//============================================================================//
//    FCUDA
//    Copyright (c) <2016> 
//    <University of Illinois at Urbana-Champaign>
//    <University of California at Los Angeles> 
//    All rights reserved.
// 
//    Developed by:
// 
//        <ES CAD Group & IMPACT Research Group>
//            <University of Illinois at Urbana-Champaign>
//            <http://dchen.ece.illinois.edu/>
//            <http://impact.crhc.illinois.edu/>
// 
//        <VAST Laboratory>
//            <University of California at Los Angeles>
//            <http://vast.cs.ucla.edu/>
// 
//        <Hardware Research Group>
//            <Advanced Digital Sciences Center>
//            <http://adsc.illinois.edu/>
//============================================================================//

package fcuda;

import cetus.analysis.*;
import cetus.hir.*;
import cetus.transforms.*;
import cetus.exec.*;

import fcuda.analysis.*;
import fcuda.ir.*;
import fcuda.transforms.*;
import fcuda.common.*;
import fcuda.utils.*;

/**
 * Implements the command line parser and controls pass ordering.
 * Users may extend this class by overriding runPasses
 * (which provides a default sequence of passes).  The derived
 * class should pass an instance of itself to the run method.
 * Derived classes have access to a protected {@link Program Program} object.
 */
public class DriverFCUDA extends Driver
{

  /**
   * Runs analysis and optimization passes on the program.
   */
  public void runPasses()
  {
    // Use option dependences/prerequisites to implement different flows for MCUDA/FCUDA/RIGEL etc.
    // (Copied from Cetus Driver)
    // In each set of option strings, the first option requires the
    // rest of the options to be set for it to run effectively
    // Options that depend on options with prerequisites should be listed before their
    // dependend options (e.g. Rigel is listed before Mcuda)
    String[][] pass_prerequisites = {
      { "wrapper", "param_core"}
    };
    for (int i = 0; i < pass_prerequisites.length; ++i) {
      if (getOptionValue(pass_prerequisites[i][0]) != null) {
        for (int j = 1; j < pass_prerequisites[i].length; ++j) {
          if (getOptionValue(pass_prerequisites[i][j]) == null) {
            System.out.println("WARNING: " + pass_prerequisites[i][0] + " flag is set but " + pass_prerequisites[i][j] + " is not set");
            System.out.println("WARNING: turning on " + pass_prerequisites[i][j]);
            setOptionValue(pass_prerequisites[i][j], "1");
          }
        }
      }
    }
    // Start passes

    System.out.println("\n*** Before Any Passes  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");


    /* Link IDExpression => Symbol object for faster future access. */
    SymbolTools.linkSymbol(program);

    TransformPass.run(new AnnotationParser(program));
    System.out.println("\n*** After AnnotationParser  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    TransformPass.run(new SingleDeclarator(program));
    System.out.println("\n*** After SingleDeclarator  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    System.out.println("\n*** After InlineDeviceFunctions  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    TransformPass.run(new SeparateInitializers(program));
    System.out.println("\n*** After SeparateInitializers  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    TransformPass.run(new AnsiDeclarations(program));
    System.out.println("\n*** After AnsiDeclarations  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");


    FCUDAGlobalData.initialize();

    // Leverage stream pragmas to handle constant memory arrays
    TransformPass.run(new StreamInsertion(program));
    System.out.println("\n*** After StreamInsertion  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    if (getOptionValue("wrapper") == null) {
      TransformPass.run(new FixFCUDAMemParams(program));
      System.out.println("\n*** After FixFCUDAMemParams ***");
      System.out.println(program);
      System.out.println("===========================================");
    }

    // Replace threadIdx dependent loops by eternal while loops and if statements
    TransformPass.run(new RemoveThrDepLoops(program));
    System.out.println("\n*** After RemoveThrDepLoops  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    // Convert compute-task scalars used ouside of compute to arrays
    TransformPass.run(new MakeArraysInCompute(program));
    System.out.println("\n*** After MakeArraysInCompute  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    // Leverage COMPUTE and TRANSFER pragmas to identify and separate the corresponding tasks
    TransformPass.run(new SplitFcudaTasks(program));
    System.out.println("\n*** After SplitFcudaTasks  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    // Tidy up variable declarations within kernel and task procedures
    TransformPass.run(new CleanKernelDecls(program));
    System.out.println("\n*** After CleanKernelDecls  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    TransformPass.run(new SerializeThreads(program));
    System.out.println("\n*** After SerializeThreads  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    TransformPass.run(new EnforceSyncs(program));
    System.out.println("\n*** After EnforceSyncs  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    TransformPass.run(new PrivatizeScalarsInThreadLoops(program));
    System.out.println("\n*** After PrivatizeScalarsInThreadLoop  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    // Do ThreadLoop unrolling and Memory partitioning
    TransformPass.run(new UnrollThreadLoops(program));
    System.out.println("\n*** After UnrollThreadLoops  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    TransformPass.run(new PartitionArrays(program));
    System.out.println("\n*** After PartitionArrays  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");


    TransformPass.run(new IfSplitPass(program));
    System.out.println("\n*** After IfSplitPass  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    TransformPass.run(new WrapBlockIdxLoop(program));
    System.out.println("\n*** After WrapBlockIdxLoop  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    TransformPass.run(new PipelineFCUDACores(program));
    System.out.println("\n*** After PipelineFCUDACores  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");


    TransformPass.run(new DuplicateForFCUDA(program));
    System.out.println("\n*** After DuplicateForFCUDA  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    // Remove threadIdx independent statements from thread-loops
    // and remove empty thread-loops
    TransformPass.run(new CleanThreadLoops(program));
    System.out.println("\n*** After CleanThreadLoops  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    TransformPass.run(new KernelStateTransform(program));
    System.out.println("\n*** After KernelStateTransform  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    TransformPass.run(new CleanSyncFunc(program));
    System.out.println("\n*** After CleanSyncFunc  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    if (getOptionValue("wrapper") != null) {
      TransformPass.run(new GenWrapperSingleKernel(program));
      System.out.println("\n*** After GenWrapperSingleKernel  ***");
      System.out.println(program.toString());
      System.out.println("===========================================");

      TransformPass.run(new FixFCUDAMemParams(program));
      System.out.println("\n*** After FixFCUDAMemParams ***");
      System.out.println(program);
      System.out.println("===========================================");

      TransformPass.run(new GenWrapperMultiKernels(program));
      System.out.println("\n*** After GenWrapperMultiKernels  ***");
      System.out.println(program.toString());
      System.out.println("===========================================");
    }

    TransformPass.run(new AddHLSPragmas(program));
    System.out.println("\n*** After AddHLSPragmas  ***");
    System.out.println(program.toString());
    System.out.println("===========================================");

    if (getOptionValue("param_core") == null) {
      TransformPass.run(new DuplicateForFCUDAByCloning(program));
      System.out.println("\n*** After DuplicateForFCUDA  ***");
      System.out.println(program.toString());
      System.out.println("===========================================");
    }

    CleanLaunches.run(program);
    ClearCUDASpecs.run(program);
  }

  public DriverFCUDA()
  {
    super();
    options.add("Fcuda", "Enable Fcuda transformations");
    options.add("param_core", "Generalize core by adding num_cores (number of cores) and core_id (core identifier)");
    options.add("wrapper", "Generate wrapper functions of kernels");
  }
  /**
   * Entry point for Cetus; creates a new Driver object,
   * and calls run on it with args.
   *
   * @param args Command line options.
   */
  public static void main(String[] args)
  {
    (new DriverFCUDA()).run(args);
  }
}

