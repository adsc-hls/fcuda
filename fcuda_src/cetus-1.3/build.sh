#!/bin/sh
# build.sh - build script for Cetus.
#

# Specify the location of the antlr.jar file for your system.
ANTLR=$(pwd)/lib/antlr.jar
if [ ! -f $ANTLR ]; then
  echo "Please define ANTLR in $0"
  exit 1
fi

# Check for java/javac/jar.
for tool in java javac jar; do
  which $tool >/dev/null
  if [ $? -ne 0 ]; then
    echo $tool is not found.
    exit 1
  fi
done

# No change is required for these variables.
CETUSROOT=$PWD
SRC="$CETUSROOT/src/*/*/*.java $CETUSROOT/src/*/*/*/*.java"
PARSER="$CETUSROOT/src/cetus/base/grammars"
# Source files for parser construction.
parser_src="
  CetusCParser.java
  CToken.java
  LineObject.java
  NewCParser.g
  Pragma.java
  Pre.g
  PreprocessorInfoChannel.java
"

case "$1" in
  parser)
  echo "Compiling the parser using ANTLR..."
  rm -rf $PARSER/*.smap $PARSER/NewCLexer.java $PARSER/NewCParser.java $PARSER/NEWCTokenTypes.java $PARSER/NEWCTokenTypes.txt
  rm -rf $PARSER/PreCParser.java $PARSER/PreCLexer.java $PARSER/PreCParserTokenTypes.java $PARSER/PreCParserTokenTypes.txt
  #Build the parser
  cd $PARSER
  java -cp $ANTLR:class antlr.Tool $PARSER/Pre.g
  java -cp $ANTLR:class antlr.Tool $PARSER/NewCParser.g
  cd -
  ;;
  compile)
  echo "Compiling the source files..."
  [ -f $PARSER/NewCParser.java ] || $0 parser
  [ -d class ] || mkdir class
  javac -g -cp $ANTLR:class -d class $SRC
  ;;
  jar)
  $0 compile $ANTLR
  echo "Archiving the class files..."
  [ -d lib ] || mkdir lib
  jar cf lib/cetus.jar -C class .
  ;;
  javadoc)
  echo "Generating JAVA documents..."
  javadoc -d api cetus $SRC
  ;;
  clean)
  echo "Cleaning up..."
  rm -rf class bin lib/cetus.jar
  ;;
  bin)
  $0 jar $ANTLR
  echo "Generating a wrapper..."
  [ -d bin ] || mkdir bin
  classpath="$ANTLR:$PWD/lib/cetus.jar"
  cat > bin/cetus << EOF
#!/bin/sh
# This file was generated automatically by build.sh.
java -cp $classpath -Xmx1g cetus.exec.Driver \$*
EOF
  chmod 755 bin/cetus
  ;;
  *)
  echo "Usage: $0 <target>"
  echo "  <target> is one of the followings:"
  echo "  bin     - compile, jar, and generate a wrapper script."
  echo "  compile - compile the source files."
  echo "  clean   - remove classes, jar, and the wrapper."
  echo "  jar     - archive the class files."
  echo "  javadoc - generate api documents."
  echo "  parser  - rebuild the parser."
  exit 1
  ;;
esac
