#!/bin/sh
JAVABIN=/usr/bin
SRC="fcuda/*.java fcuda/*/*.java"


if [ ! -d jar ]; 
then
	mkdir jar
	cp cetus-1.3/lib/antlr.jar jar/
fi

case "$1" in
  compile)
  echo "Compiling the Cetus 1.3 source files..."
  cd cetus-1.3
  ./build.sh jar
  cp lib/cetus.jar ../jar
  cd ..
  echo "Compiling the FCUDA source files..."
  cd src
  [ -d class ] || mkdir class
  $JAVABIN/javac -g -classpath ../jar/cetus.jar:../jar/antlr.jar -d class $SRC
  echo "Archiving the class files..."
  cd ..
  $JAVABIN/jar cf jar/fcuda.jar -C src/class .
  ;;

  clean)
  echo "Cleaning up..."
  rm -rf src/class
  rm -rf cetus-1.3/class
  rm -rf jar/cetus.jar jar/fcuda.jar
  ;;
  *)
  echo "Usage: $0 {compile|clean}"
  echo "  compile - compile source files"
  echo "  clean   - clean binary files"
  exit 1
  ;;
esac
