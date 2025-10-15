"""
Logging infrastructure for the Invoice Extractor
Provides structured logging with file and console output
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_dir='logs',
    log_level=logging.INFO,
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5,
    enable_console=True,
    enable_file=True
):
    """
    Setup logging configuration for the application
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        enable_console: Enable console logging
        enable_file: Enable file logging
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler - detailed logs
    if enable_file:
        log_file = log_path / f"invoice_extractor_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Console handler - simple logs with colors
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        if sys.stdout.isatty():
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # Error file handler - separate file for errors
    if enable_file:
        error_log_file = log_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    logger.info("="*80)
    logger.info("Logging initialized")
    logger.info(f"Log directory: {log_path.absolute()}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info("="*80)


def get_logger(name):
    """
    Get a logger instance with the given name
    
    Args:
        name: Name of the logger (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporary log level changes"""
    
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.old_level = None
    
    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_function_call(func):
    """Decorator to log function calls with arguments and return values"""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper


def log_execution_time(func):
    """Decorator to log function execution time"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {elapsed_time:.2f}s")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed_time:.2f}s: {e}")
            raise
    return wrapper