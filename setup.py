#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from pathlib import Path

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()


def strip_comments(l):
    return l.split('#', 1)[0].strip()


def _pip_requirement(req, *root):
    if req.startswith('-r '):
        _, path = req.split()
        return reqs(*root, *path.split('/'))
    return [req]


def _reqs(*f):
    path = (Path.cwd()).joinpath(*f)
    with path.open() as fh:
        reqs = [strip_comments(l) for l in fh.readlines()]
        return [_pip_requirement(r, *f[:-1]) for r in reqs if r]


def reqs(*f):
    return [req for subreq in _reqs(*f) for req in subreq]


install_requires = reqs('requirements.txt')
test_requires = install_requires

setup(
    author="Giovanni Marelli",
    author_email='adyanov@protonmail.com',
    # classifiers=[
    #     'Development Status :: 2 - Pre-Alpha',
    #     'Intended Audience :: Developers',
    #     'Programming Language :: Python :: 3.6',
    #     'Programming Language :: Python :: 3.7',
    # ],
    description="paiting with machine learning",
    # entry_points={
    #     'console_scripts': [
    #         'albio=albio.main:cli',
    #     ],
    # },
    # long_description=readme + '\n\n' + history,
    # include_package_data=True,
    keywords='ml_paint',
    name='ml_paint',
    setup_requires=install_requires,
    install_requires=install_requires,
    # test_suite='tests',
    packages=find_packages(),
    tests_require=test_requires,
    url='https://github.com/sabeiro/ml_paint',
    version='0.1.0',
    # zip_safe=False,
)
