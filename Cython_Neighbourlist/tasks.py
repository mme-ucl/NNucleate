""" Task definitions for invoke command line utility for python bindings
    overview article.
"""
import invoke
import pathlib
import sys
import os
import shutil
import re
import glob



def print_banner(msg):
    print("==================================================")
    print("= {} ".format(msg))


@invoke.task()
def build_cmult(c, path=None):
    """Build the shared library for the sample C code"""
    # Moving this type hint into signature causes an error (???)
    c: invoke.Context
    if on_win:
        if not path:
            print("Path is missing")
        else:
            # Using c.cd didn't work with paths that have spaces :/
            path = f'"{path}vcvars32.bat" x86'  # Enter the VS venv
            path += f'&& cd "{os.getcwd()}"'  # Change to current dir
            path += "&& cl /LD cmult.c"  # Compile
            # Uncomment line below, to suppress stdout
            # path = path.replace("&&", " >nul &&") + " >nul"
            c.run(path)
    else:
        print_banner("Building C Library")
        cmd = "gcc -c -Wall -Werror -fpic cmult.c -I /usr/include/python3.7"
        invoke.run(cmd)
        invoke.run("gcc -shared -o libcmult.so cmult.o")
        print("* Complete")



@invoke.task()
def build_cppmult(c):
    """Build the shared library for the sample C++ code"""
    print_banner("Building C++ Library")
    invoke.run(
        "g++ -O3 -Wall -shared -std=c++11 -fPIC neighborlist.cpp "
        "-o libnl.so "
    )
    print("* Complete")


def compile_python_module(cpp_name, extension_name):
    invoke.run(
        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC "
        "`python3 -m pybind11 --includes` "
        "-I /usr/ulocal/anaconda/include/python3.9 -I .  "
        "{0} "
        "-o {1}`python3.9-config --extension-suffix` "
        "-L. -lnl -Wl,-rpath,.".format(cpp_name, extension_name)
    )



@invoke.task(build_cppmult)
def build_cython(c):
    """Build the cython extension module"""
    print_banner("Building Cython Module")
    # Run cython on the pyx file to create a .cpp file
    invoke.run("cython --cplus -3 neighborlist.pyx -o nl_wrapper.cpp")

    # Compile and link the cython wrapper library
    compile_python_module("nl_wrapper.cpp", "neighborlist")
    print("* Complete")

