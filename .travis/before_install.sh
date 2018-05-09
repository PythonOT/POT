#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on OS X
    # e.g. brew install pyenv-virtualenv
    #brew update
    #brew install python
    sudo easy_install -U pip

else
    # Install some custom requirements on Linux
    sudo apt-get update -q
    sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev
fi
