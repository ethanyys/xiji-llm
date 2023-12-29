import configparser
import json
import os

from configparser import NoOptionError, NoSectionError

__conf = configparser.ConfigParser().ConfigParser()
__conf.optionxform = str

pwd = os.getcwd()


def get_string(section, key, default, env=False):
    try:
        __conf.read(pwd + "/conf/app.cfg", "utf-8")
        if env:
            return __conf.get(section, key, vars=os.environ)
        return __conf.get(section, key)
    except NoOptionError as e:
        return default
    except NoSectionError as e:
        return default


def get_int(section, key, default: int, env=False):
    try:
        __conf.read(pwd + "/conf/app.cfg", "utf-8")
        if env:
            return __conf.getint(section, key, vars=os.environ)
        return __conf.getint(section, key)
    except NoOptionError as e:
        return default
    except NoSectionError as e:
        return default


def get_float(section, key, default: float, env=False):
    try:
        __conf.read(pwd + "/conf/app.cfg", "utf-8")
        if env:
            return __conf.getfloat(section, key, vars=os.environ)
        return __conf.getfloat(section, key)
    except NoOptionError as e:
        return default
    except NoSectionError as e:
        return default


def get_boolean(section, key, default: bool, env=False):
    try:
        __conf.read(pwd + "/conf/app.cfg", "utf-8")
        if env:
            return __conf.getboolean(section, key, vars=os.environ)
        return __conf.getboolean(section, key)
    except NoOptionError as e:
        return default
    except NoSectionError as e:
        return default


def get_list(section, key, default, env=False):
    try:
        __conf.read(pwd + "/conf/app.cfg", "utf-8")
        if env:
            raise ValueError("list config do not support reading from env")
        return json.loads(__conf.get(section, key))
    except NoOptionError as e:
        return default
    except NoSectionError as e:
        return default
