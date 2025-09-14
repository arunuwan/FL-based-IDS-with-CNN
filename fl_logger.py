import json
import os
import time
from datetime import datetime
import numpy as np

class FLLogger:
    """Logger for Federated Learning monitoring data"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.ensure_log_dir()
    
    def ensure_log_dir(self):
        """Create logs directory if it doesn't exist"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def log_training_step(self, server_name, epoch, accuracy, loss, val_accuracy=None, val_loss=None):
        """Log training step for a specific server"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'server': server_name,
            'epoch': epoch,
            'accuracy': float(accuracy),
            'loss': float(loss),
            'val_accuracy': float(val_accuracy) if val_accuracy is not None else None,
            'val_loss': float(val_loss) if val_loss is not None else None
        }
        
        # Save to individual server log file
        server_log_file = os.path.join(self.log_dir, f"{server_name.lower()}_training.json")
        self._append_to_json_file(server_log_file, log_entry)
        
        # Save to combined log file
        combined_log_file = os.path.join(self.log_dir, "combined_training.json")
        self._append_to_json_file(combined_log_file, log_entry)
    
    def log_federated_aggregation(self, epoch, test_accuracy, test_loss, server_count):
        """Log federated aggregation results"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'federated_aggregation',
            'epoch': epoch,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'server_count': server_count
        }
        
        # Save to federated log file
        fed_log_file = os.path.join(self.log_dir, "federated_aggregation.json")
        self._append_to_json_file(fed_log_file, log_entry)
    
    def log_system_status(self, status, message=""):
        """Log system status updates"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'system_status',
            'status': status,
            'message': message
        }
        
        # Save to system log file
        system_log_file = os.path.join(self.log_dir, "system_status.json")
        self._append_to_json_file(system_log_file, log_entry)
    
    def _append_to_json_file(self, file_path, log_entry):
        """Append log entry to JSON file"""
        try:
            # Load existing data
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Append new entry
            data.append(log_entry)
            
            # Save back to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error logging to {file_path}: {e}")
    
    def get_latest_metrics(self):
        """Get latest metrics from all logs"""
        metrics = {}
        
        # Get latest from each server
        for server in ['server1', 'server2', 'server3', 'server4', 'server5']:
            server_file = os.path.join(self.log_dir, f"{server}_training.json")
            if os.path.exists(server_file):
                try:
                    with open(server_file, 'r') as f:
                        data = json.load(f)
                        if data:
                            latest = data[-1]
                            metrics[server] = {
                                'accuracy': latest.get('accuracy', 0),
                                'loss': latest.get('loss', 0),
                                'epoch': latest.get('epoch', 0)
                            }
                except:
                    continue
        
        return metrics

# Global logger instance
fl_logger = FLLogger()
