from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


keywords = ("attitude determination")

setup(
    name="attitude_determination",
    version="1.0.0",
    description="Attitude Determination Testing",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url="",
    author="Guilherme Henrique dos Santos",
    author_email="dos_santos_98@hotmail.com",
    keywords=keywords,
    license="MIT",
    packages=find_packages(exclude=['tests', 'models']),
    install_requires=['numpy>= 1.18',
                      'pandas >= 1.0.3',
                      'scipy >= 1.4.1',
                      'opencv-python >= 4.2.0'],
    python_requires='>=3.6',
)