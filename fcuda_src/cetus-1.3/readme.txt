--------------------------------------------------------------------------------
RELEASE
--------------------------------------------------------------------------------
Cetus 1.3 (June 13, 2011)

Cetus is a source-to-source compiler infrastructure for C written in Java, and
can be downloaded from http://cetus.ecn.purdue.edu. This version contains
improvements to various analysis/transformation passes, both in their
effectiveness and efficiency.
As a side note, we are preparing for Cetus Users and Compiler Infrastructure
Workshop in October -- visit http://cetus.ecn.purdue.edu/cetusworkshop for more
information.

--------------------------------------------------------------------------------
FEATURES/UPDATES
--------------------------------------------------------------------------------
* Bug fixes
  - Points-to analysis produces the same result over repeated invocations while
    the previous version did not.
  - Added checks in LoopProfiler to avoid profiling loops that contain jumps to
    program points other than the loop exits.
  - Fixed the OpenMP code generator so that it excludes variables declared
    within the relevant loop from the private clause.
  - Fixed the expression simplifier's behavior that produced incorrect results
    on logical expressions.
  - Fixed the dependence analyzer's behavior that produced incorrect results on
    certain scalar variables which contain pointer dereferences.
  - Fixed a bug that prevented selecting parallel loops for OpenMP
    parallelization in certain cases.

* New analysis/transform passes
  - Inline expansion
    This pass has been mature since the last release. Various sub-options
    provide configurable behaviors of the inline transformation.
  - Unreachable branch elimination
    This pass detects unreachable code due to branches that can be evaluated
    at compile time and eliminates such code. The tested branches are in IF,
    FOR, WHILE, DO, and SWITCH statements. See the "-teliminate-branch" flag to
    see how to invoke this transformation.
  - Reduction transform
    This pass produces OpenMP-conforming code from the reduction items
    recognized by the reduction analysis in Cetus. Two types of transformations
    are performed; one is scalar replacement for loop-invariant expressions, and
    the other is array reduction code generation. This transformation pass is
    invoked by adjusting the value of the flag "-reduction".
  - Def-Use chains and its interface
    This analysis computes def-use chains and use-def chains
    interprocedurally for a program. The analysis returns a list of
    statements from a given def or use expression. The analysis is located
    in the package "cetus.application". The interface, DefUseChain and
    UseDefChain, and its implementation, IPChainAnalysis are the key classes
    for this feature.    

* Updates to automatic parallelization
  - Enhanced report of automatic parallelization
    We added options that enables detailed reporting of automatic
    parallelization. The report shows variables that may carry data dependences
    if the enclosing loop is not parallelized.
    This reporting is turned on when users pass the following flags:
      -parallelize-loops=3
        Every loop is analyzed for automatic parallelization but only the
        outermost parallel loops are scheduled for parallelization (OpenMP)
      -parallelize-loops=4
        Every loop is analyzed for automatic parallelization and every parallel
        loop is scheduled for parallelization
  - Enhanced handling of known library calls
    Handling of library calls in analysis passes has been improved to produce
    less conservative analysis results. We inserted comprehensive check routines
    in range analysis, array privatization pass, and loop parallelization pass
    so that they identify side-effect-free standard library calls and make less
    conservative decisions. This behavior is unsafe when the input program
    contains a call to a non-standard-library function that matches the
    signature of a standard-library function.
  - Enhanced handling of reductions
    The previous version was not properly handling reduction items that do not
    conform to the OpenMP specification. A new pass was added and called by the
    loop parallelization pass to overcome this limitation.
    See the "ReductionTransform" pass for more details.
  - Enhanced OpenMP code generation with a simple performance model
    Cetus' OpenMP code generator now inserts simple run-time checks through
    OpenMP IF clauses so that it serialize parallel loops that contain
    insufficient work.
  - As a result, the current version finds 21 more parallel loops than the
    previous version does for our test suite (NPB, SPEC OMP).

* Updates for efficiency and consistency
  We found that Cetus suffers from limited scalability in terms of memory usage 
  and put significant effort into improving Cetus' memory usage. This
  optimization includes minimization of object clones, use of in-place
  operations if possible, use of efficient collections, and other optimizations
  for time efficiency.
  The following numbers summarize the result of this optimization:
    Improvements since the last release (1.2.1)
    Environments: Quad-core AMD Opteron at 2.5GHz
                  JAVA SE 1.6.0_25 with -Xmx=1500m
    Elapsed time for building and printing IRs for 25 NPB/SPEC codes
        1.2.1 : 709 seconds
        1.3   : 242 seconds
    Elapsed time for automatic parallelization for 11 NPB/SPEC codes
        1.2.1 : 143 seconds
        1.3   : 110 seconds

* Updates in flags
  - New flags
    -debug_parser_input
        Print a single preprocessed input file before sending it to the parser,
        and exit
    -debug_preprocessor_input
        Print a single pre-annotated input file before sending it to the
        preprocessor, and exit
    -dump-options
        Create file options.cetus with default options
    -dump-system-options
        Create a system wide file options.cetus with default options
    -parser=PARSERNAME
        Name of the parser to be used for parsing the source files
    -teliminate-branch=N
        Eliminates unreachable branch targets
    -tinline=SUBOPTION
        This flags replaces "-tinline-expansion"
    -profitable-omp=N
        Inserts runtime checks for selecting profitable omp parallel region
  - Modified flags
    -ddt=N
        ddt=3 was removed from the help message
    -parallelize-loops=N
        By default, only outermost loops are parallelized
        Added options for printing parallelization report
    -profile-loops=N
        Clarified the types of profiled regions -- loop, OpenMP parallel region,
        and OpenMP for loop
  - Removed flags
    -tinline-expansion
        -tinline replaces this flag

--------------------------------------------------------------------------------
CONTENTS
--------------------------------------------------------------------------------
This Cetus release has the following contents.

  lib            - Archived classes (jar)
  license.txt    - Cetus license
  build.sh       - Command line build script
  build.xml      - Build configuration for Apache Ant
  src            - Cetus source code
  readme.txt     - This file
  readme_log.txt - Archived release notes
  readme_omp2gpu.txt - readme file for OpenMP-to-CUDA translator

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
* JAVA SE 6
* ANTLRv2 
* GCC
 
--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------
* Obtain Cetus distribution
  The latest version of Cetus can be obtained at:
  http://cetus.ecn.purdue.edu/

* Unpack
  Users need to unpack the distribution before installing Cetus.
  $ cd <directory_where_cetus.tar.gz_exists>
  $ gzip -d cetus.tar.gz | tar xvf -

* Build
  There are several options for building Cetus:
  - For Apache Ant users
    The provided build.xml defines the build targets for Cetus. The available
    targets are "compile", "jar", "clean" and "javadoc". Users need to edit
    the location of the Antlr tool.
  - For Linux/Unix command line users.
    Run the script build.sh after defining system-dependent variables in the
    script.
  - For SDK (Eclipse, Netbeans, etc) users
    Follow the instructions of each SDK.

--------------------------------------------------------------------------------
RUNNING CETUS
--------------------------------------------------------------------------------
Users can run Cetus in the following way:

  $ java -classpath=<user_class_path> cetus.exec.Driver <options> <C files>

The "user_class_path" should include the class paths of Antlr and Cetus.
"build.sh" and "build.xml" provides a target that generates a wrapper script
for Cetus users.

--------------------------------------------------------------------------------
TESTING
--------------------------------------------------------------------------------
We have tested Cetus successfully using the following benchmark suites:

* SPEC CPU2006
  More information about this suite is available at http://www.spec.org

* SPEC OMP2001
  More information about this suite is available at http://www.spec.org

* NPB 2.3 written in C
  More information about this suite is available at
  http://www.hpcs.cs.tsukuba.ac.jp/omni-openmp/

June 13, 2011
The Cetus Team

URL: http://cetus.ecn.purdue.edu
EMAIL: cetus@ecn.purdue.edu
