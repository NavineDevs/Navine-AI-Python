from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys


class BuildExtWithOptional(build_ext):
    def run(self):
        try:
            super().run()
        except Exception:
            print("WARNING: native extensions failed to build, falling back to pure Python")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception:
            print(f"WARNING: extension {ext.name} failed to build, skipping")


def get_extensions():
    extensions = []
    try:
        from torch.utils.cpp_extension import CppExtension

        extensions.append(
            CppExtension(
                "navine.engine.kernels._attention_kernel",
                [os.path.join("navine", "engine", "kernels", "attention_kernel.cpp")],
            )
        )
    except ImportError:
        pass
    return extensions


setup(
    name="navine-ai",
    version="0.1.0",
    packages=find_packages(include=["navine*"]),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtWithOptional},
)
