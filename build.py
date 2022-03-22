"""Build script."""

import shutil
from distutils import log as distutils_log
from pathlib import Path
from typing import Any, Dict
import skbuild
import skbuild.constants
import setuptools
import glob
import sys
import distutils
from setuptools import Extension
from distutils.command.build_ext import build_ext


class build_ext_check_gcc(build_ext):
    def build_extensions(self):
        c = self.compiler
        _compile = c._compile

        def c_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            cc_args = cc_args + ["-std=c99"] if src.endswith(".c") else cc_args
            return _compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        if c.compiler_type == "unix" and "gcc" in c.compiler:
            c._compile = c_compile
        elif self.compiler.compiler_type == "msvc":
            if sys.version_info[:2] < (3, 5):
                c.include_dirs.extend(["crfsuite/win32"])
        build_ext.build_extensions(self)


def build_files(dest_dir, src_dir):
    remove_files(dest_dir, "**/*.so")
    copy_files(src_dir, dest_dir, "**/*.so")
    remove_files(dest_dir, "**/*.pyd")
    copy_files(src_dir, dest_dir, "**/*.pyd")
    
def build(setup_kwargs: Dict[str, Any]) -> None:
    """Build Levenshtein extensions."""
    skbuild.setup(
        **setup_kwargs, 
        script_args=["build_ext"],
        packages=['dataprep'])
    cmake_src_dir = Path(skbuild.constants.CMAKE_INSTALL_DIR()) / "src" / "Levenshtein"
    cmake_dest_dir = Path("extern/Levenshtein")
    build_files(cmake_dest_dir, cmake_src_dir)

    """Build python-crfsuite extension"""
    src_dir = Path("extern/python-crfsuite")
    sources = [f"{src_dir}/pycrfsuite/_pycrfsuite.cpp", f"{src_dir}/pycrfsuite/trainer_wrapper.cpp"]
    # crfsuite
    sources += glob.glob(f"{src_dir}/crfsuite/lib/crf/src/*.c")
    sources += glob.glob(f"{src_dir}/crfsuite/swig/*.cpp")
    sources += [f"{src_dir}/crfsuite/lib/cqdb/src/cqdb.c"]
    sources += [f"{src_dir}/crfsuite/lib/cqdb/src/lookup3.c"]
    # lbfgs
    sources += glob.glob(f"{src_dir}/liblbfgs/lib/*.c")

    includes = [
        f"{src_dir}/crfsuite/include/",
        f"{src_dir}/crfsuite/lib/cqdb/include",
        f"{src_dir}/liblbfgs/include",
        f"{src_dir}/pycrfsuite"
    ]
    ext_modules = [
        Extension(
            name = "pycrfsuite._pycrfsuite", 
            include_dirs=includes, 
            language="c++", 
            sources=sorted(sources)
        )
    ]
    setup_kwargs.update(
        {"ext_modules": ext_modules, "cmdclass": {"build_ext": build_ext_check_gcc}}
    )
    setuptools.setup(**setup_kwargs, script_args=["build_ext"])
    pathes = glob.glob("build/**")
    for path in pathes:
        build_src_dir = Path(path)
        dest_dir = Path("extern/python-crfsuite")
        build_files(dest_dir, build_src_dir)


def remove_files(target_dir: Path, pattern: str) -> None:
    """Delete files matched with a glob pattern in a directory tree."""
    for path in target_dir.glob(pattern):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        distutils_log.info(f"removed {path}")  # type: ignore[call-arg]
        # will be fixed in mypy 0.800, https://github.com/python/typeshed/pull/4573


def copy_files(src_dir: Path, dest_dir: Path, pattern: str) -> None:
    """Copy files matched with a glob pattern in a directory tree to another."""
    for src in src_dir.glob(pattern):
        dest = dest_dir / src.relative_to(src_dir)
        if src.is_dir():
            # NOTE: inefficient if subdirectories also match to the pattern.
            copy_files(src, dest, "*")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            distutils_log.info(f"copied {src} to {dest}")  # type: ignore[call-arg]
            # will be fixed in mypy 0.800, https://github.com/python/typeshed/pull/4573


if __name__ == "__main__":
    build({})
    
