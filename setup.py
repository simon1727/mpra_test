#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 
    'pyyaml==6.0.1',
    'pandas==2.2.2',
    'numpy==1.26.4',
    'torch',
]

test_requirements = ['Click>=7.0', 
    'pyyaml==6.0.1',
    'pandas==2.2.2',
    'numpy==1.26.4', ]

setup(
    author="Yilun Sheng",
    author_email='simon1727@qq.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    description="MPRA Test.",
    entry_points={
        'console_scripts': [
            'mpra_test=mpra_test.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mpra_test',
    name='mpra_test',
    packages=find_packages(include=['mpra_test', 'mpra_test.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/simon1727/mpra_test',
    version='0.1.0',
    zip_safe=False,
)
