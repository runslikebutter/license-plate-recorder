"""
S3 Uploader for License Plate Recordings
Handles asynchronous upload to S3-compatible storage (Backblaze B2) with local file cleanup
"""

import os
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from utils.logger import setup_logger


class S3UploadStatus:
    """Status tracking for uploads"""
    PENDING = "pending"
    UPLOADING = "uploading"
    SUCCESS = "success"
    FAILED = "failed"


class S3Uploader:
    """
    Asynchronous S3 uploader for recording files
    Uploads files in background thread and deletes local files after upload
    """

    def __init__(self,
                 bucket_name: str,
                 endpoint_url: str,
                 access_key_id: str,
                 secret_access_key: str,
                 enabled: bool = True,
                 delete_after_upload: bool = True,
                 max_retries: int = 3,
                 retry_delay: float = 5.0,
                 on_upload_complete: Optional[Callable] = None,
                 on_upload_failed: Optional[Callable] = None):
        """
        Initialize S3 uploader

        Args:
            bucket_name: S3 bucket name
            endpoint_url: S3 endpoint URL (for Backblaze B2: s3.us-east-005.backblazeb2.com)
            access_key_id: S3 access key ID
            secret_access_key: S3 secret access key
            enabled: Enable/disable uploader
            delete_after_upload: Delete local file after successful upload
            max_retries: Maximum upload retry attempts
            retry_delay: Delay between retries in seconds
            on_upload_complete: Callback function when upload completes (file_path, s3_key)
            on_upload_failed: Callback function when upload fails (file_path, error)
        """
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.enabled = enabled
        self.delete_after_upload = delete_after_upload
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.on_upload_complete = on_upload_complete
        self.on_upload_failed = on_upload_failed

        self.logger = setup_logger(self.__class__.__name__)

        # Upload queue and tracking
        self.upload_queue = Queue()
        self.upload_status: Dict[str, Dict[str, Any]] = {}
        self.status_lock = threading.Lock()

        # Worker thread
        self.worker_thread = None
        self.running = False

        # Statistics
        self.stats = {
            'total_queued': 0,
            'total_uploaded': 0,
            'total_failed': 0,
            'total_bytes_uploaded': 0,
            'total_deleted': 0
        }
        self.stats_lock = threading.Lock()

        # Initialize S3 client if enabled
        self.s3_client = None
        if self.enabled:
            self._initialize_s3_client()
        else:
            self.logger.info("S3 uploader is DISABLED")

    def _initialize_s3_client(self):
        """Initialize boto3 S3 client"""
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=f'https://{self.endpoint_url}',
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key
            )
            
            # Test connection by checking if bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.info(f"S3 client initialized successfully: {self.bucket_name} @ {self.endpoint_url}")
            
        except NoCredentialsError:
            self.logger.error("S3 credentials not found")
            self.enabled = False
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                self.logger.error(f"S3 bucket not found: {self.bucket_name}")
            elif error_code == '403':
                self.logger.error(f"Access denied to S3 bucket: {self.bucket_name}")
            else:
                self.logger.error(f"S3 client initialization error: {e}")
            self.enabled = False
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {e}")
            self.enabled = False

    def start(self):
        """Start the upload worker thread"""
        if not self.enabled:
            self.logger.warning("S3 uploader not started - disabled")
            return

        if self.running:
            self.logger.warning("S3 uploader already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.worker_thread.start()
        self.logger.info("S3 upload worker started")

    def stop(self, timeout: float = 30.0, wait_for_uploads: bool = True):
        """
        Stop the upload worker thread
        
        Args:
            timeout: Maximum time to wait for pending uploads to complete
            wait_for_uploads: If True, wait for pending uploads to finish
        """
        if not self.running:
            return

        pending = self.upload_queue.qsize()
        if pending > 0 and wait_for_uploads:
            self.logger.info(f"Stopping S3 uploader... waiting for {pending} pending upload(s)")
        else:
            self.logger.info("Stopping S3 uploader...")
        
        self.running = False

        if self.worker_thread:
            self.worker_thread.join(timeout=timeout)
            if self.worker_thread.is_alive():
                remaining = self.upload_queue.qsize()
                self.logger.warning(
                    f"Upload worker did not stop within {timeout}s timeout. "
                    f"Remaining items in queue: {remaining}"
                )
            else:
                remaining = self.upload_queue.qsize()
                if remaining > 0:
                    self.logger.warning(
                        f"Upload worker stopped with {remaining} item(s) still in queue"
                    )
                else:
                    self.logger.info("Upload worker stopped cleanly")

        self._log_statistics()

    def queue_upload(self, file_path: str, s3_prefix: str = "recordings") -> bool:
        """
        Queue a file for upload

        Args:
            file_path: Path to local file to upload
            s3_prefix: S3 key prefix (folder path)

        Returns:
            True if queued successfully
        """
        if not self.enabled:
            self.logger.debug(f"Upload disabled, skipping: {file_path}")
            # Still delete file if configured (even when S3 unavailable)
            if self.delete_after_upload and os.path.exists(file_path):
                if self._delete_local_file(file_path):
                    with self.stats_lock:
                        self.stats['total_deleted'] += 1
            return False

        if not os.path.exists(file_path):
            self.logger.error(f"File not found, cannot queue: {file_path}")
            return False

        file_size = os.path.getsize(file_path)
        filename = os.path.basename(file_path)
        s3_key = f"{s3_prefix}/{filename}" if s3_prefix else filename

        # Add to upload status tracking
        with self.status_lock:
            self.upload_status[file_path] = {
                'status': S3UploadStatus.PENDING,
                'file_path': file_path,
                's3_key': s3_key,
                'file_size': file_size,
                'queued_time': time.time(),
                'upload_start_time': None,
                'upload_end_time': None,
                'error': None,
                'retries': 0
            }

        # Add to queue
        self.upload_queue.put({
            'file_path': file_path,
            's3_key': s3_key,
            'file_size': file_size
        })

        with self.stats_lock:
            self.stats['total_queued'] += 1

        self.logger.info(f"Queued for upload: {filename} ({file_size} bytes) -> s3://{self.bucket_name}/{s3_key}")
        return True

    def _upload_worker(self):
        """Worker thread that processes upload queue"""
        self.logger.info("Upload worker started")

        while self.running:
            try:
                # Get item from queue with timeout (allows checking self.running periodically)
                upload_item = self.upload_queue.get(timeout=1.0)
                
                file_path = upload_item['file_path']
                s3_key = upload_item['s3_key']
                file_size = upload_item['file_size']

                # Check if we're shutting down - still process this item
                if not self.running:
                    self.logger.info(f"Processing final upload before shutdown: {os.path.basename(file_path)}")

                # Update status
                with self.status_lock:
                    if file_path in self.upload_status:
                        self.upload_status[file_path]['status'] = S3UploadStatus.UPLOADING
                        self.upload_status[file_path]['upload_start_time'] = time.time()

                # Attempt upload with retries
                success = self._upload_file_with_retry(file_path, s3_key)

                # Update status
                with self.status_lock:
                    if file_path in self.upload_status:
                        self.upload_status[file_path]['upload_end_time'] = time.time()
                        self.upload_status[file_path]['status'] = (
                            S3UploadStatus.SUCCESS if success else S3UploadStatus.FAILED
                        )

                # Update statistics
                with self.stats_lock:
                    if success:
                        self.stats['total_uploaded'] += 1
                        self.stats['total_bytes_uploaded'] += file_size
                    else:
                        self.stats['total_failed'] += 1

                # Delete local file on success OR failure (after all retries exhausted)
                if self.delete_after_upload and os.path.exists(file_path):
                    if self._delete_local_file(file_path):
                        with self.stats_lock:
                            self.stats['total_deleted'] += 1

                # Callbacks
                if success and self.on_upload_complete:
                    try:
                        self.on_upload_complete(file_path, s3_key)
                    except Exception as e:
                        self.logger.error(f"Error in upload complete callback: {e}")
                elif not success and self.on_upload_failed:
                    error_msg = self.upload_status.get(file_path, {}).get('error', 'Unknown error')
                    try:
                        self.on_upload_failed(file_path, error_msg)
                    except Exception as e:
                        self.logger.error(f"Error in upload failed callback: {e}")

                self.upload_queue.task_done()

            except Empty:
                # No items in queue, continue waiting
                continue
            except Exception as e:
                self.logger.error(f"Error in upload worker: {e}")
                # Make sure to mark task as done even on error to prevent blocking
                try:
                    self.upload_queue.task_done()
                except ValueError:
                    pass  # task_done() called too many times

        # Process any remaining items in queue before exiting (if shutting down gracefully)
        remaining = self.upload_queue.qsize()
        if remaining > 0:
            self.logger.info(f"Processing {remaining} remaining upload(s) before exit...")
            while not self.upload_queue.empty():
                try:
                    upload_item = self.upload_queue.get_nowait()
                    file_path = upload_item['file_path']
                    s3_key = upload_item['s3_key']
                    
                    self.logger.info(f"Final upload: {os.path.basename(file_path)}")
                    success = self._upload_file_with_retry(file_path, s3_key)
                    
                    # Update statistics
                    with self.stats_lock:
                        if success:
                            self.stats['total_uploaded'] += 1
                            self.stats['total_bytes_uploaded'] += upload_item['file_size']
                        else:
                            self.stats['total_failed'] += 1
                    
                    # Delete file on success OR failure (after all retries)
                    if self.delete_after_upload and os.path.exists(file_path):
                        self._delete_local_file(file_path)
                        with self.stats_lock:
                            self.stats['total_deleted'] += 1
                    
                    self.upload_queue.task_done()
                    
                except Empty:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing final upload: {e}")

        self.logger.info("Upload worker stopped")

    def _upload_file_with_retry(self, file_path: str, s3_key: str) -> bool:
        """
        Upload file with retry logic

        Args:
            file_path: Local file path
            s3_key: S3 object key

        Returns:
            True if upload succeeded
        """
        for attempt in range(self.max_retries):
            try:
                if not os.path.exists(file_path):
                    self.logger.error(f"File not found for upload: {file_path}")
                    with self.status_lock:
                        if file_path in self.upload_status:
                            self.upload_status[file_path]['error'] = "File not found"
                    return False

                # Perform upload
                start_time = time.time()
                self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
                upload_time = time.time() - start_time

                file_size = os.path.getsize(file_path)
                upload_speed = file_size / upload_time if upload_time > 0 else 0

                self.logger.info(
                    f"Upload successful: {os.path.basename(file_path)} -> s3://{self.bucket_name}/{s3_key} "
                    f"({file_size} bytes in {upload_time:.2f}s, {upload_speed/1024:.1f} KB/s)"
                )
                return True

            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_msg = f"S3 error ({error_code}): {e.response['Error']['Message']}"
                
                with self.status_lock:
                    if file_path in self.upload_status:
                        self.upload_status[file_path]['retries'] = attempt + 1
                        self.upload_status[file_path]['error'] = error_msg

                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"Upload failed (attempt {attempt + 1}/{self.max_retries}): {error_msg}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Upload failed after {self.max_retries} attempts: {error_msg}")

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                
                with self.status_lock:
                    if file_path in self.upload_status:
                        self.upload_status[file_path]['retries'] = attempt + 1
                        self.upload_status[file_path]['error'] = error_msg

                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"Upload failed (attempt {attempt + 1}/{self.max_retries}): {error_msg}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Upload failed after {self.max_retries} attempts: {error_msg}")

        return False

    def _delete_local_file(self, file_path: str) -> bool:
        """
        Delete local file after upload

        Args:
            file_path: Path to file to delete

        Returns:
            True if deleted successfully
        """
        try:
            os.remove(file_path)
            self.logger.info(f"Deleted local file: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete local file {file_path}: {e}")
            return False

    def get_upload_status(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get upload status for a specific file"""
        with self.status_lock:
            return self.upload_status.get(file_path)

    def get_pending_uploads(self) -> int:
        """Get number of pending uploads"""
        return self.upload_queue.qsize()

    def get_statistics(self) -> Dict[str, Any]:
        """Get upload statistics"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        stats['pending_uploads'] = self.get_pending_uploads()
        stats['enabled'] = self.enabled
        stats['running'] = self.running
        
        return stats

    def _log_statistics(self):
        """Log upload statistics"""
        stats = self.get_statistics()
        
        self.logger.info("="*60)
        self.logger.info("S3 UPLOADER STATISTICS")
        self.logger.info(f"Enabled: {stats['enabled']}")
        self.logger.info(f"Total queued: {stats['total_queued']}")
        self.logger.info(f"Total uploaded: {stats['total_uploaded']}")
        self.logger.info(f"Total failed: {stats['total_failed']}")
        self.logger.info(f"Total bytes uploaded: {stats['total_bytes_uploaded']} ({stats['total_bytes_uploaded']/1024/1024:.2f} MB)")
        self.logger.info(f"Total files deleted: {stats['total_deleted']}")
        self.logger.info(f"Pending uploads: {stats['pending_uploads']}")
        self.logger.info("="*60)

    def is_enabled(self) -> bool:
        """Check if uploader is enabled"""
        return self.enabled

    def is_running(self) -> bool:
        """Check if worker thread is running"""
        return self.running

