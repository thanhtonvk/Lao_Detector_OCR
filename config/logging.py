from . import config


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s %(module)s :: %(message)s",
        },
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default"],
            "level": "DEBUG" if config.DEBUG else "INFO",
            "propagate": False,
        },
        # 'modules': {
        #     'handlers': ['default'],
        #     'level': 'DEBUG',
        #     'propagate': False
        # },
        # '__main__': {  # if __name__ == '__main__'
        #     'handlers': ['default'],
        #     'level': 'DEBUG',
        #     'propagate': False
        # },
    },
}
