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

package fcuda.transforms;
import fcuda.utils.*;

import java.io.*;
import java.util.*;

import cetus.hir.*;

/**
 * Interfaces a function call to the MCUDA runtime library.
 * First creates a type definition for this function's parameter 
 * struction.  
 * Then modifies the current function to now accept the implicit 
 * block and grid variables of the CUDA programming model, 
 * in addition to a pointer to an instance of the parameter structure.
 * Lastly, creates a new function with the original function 
 * signature, whose sole purpose is to construct the paramter 
 * structure, and pass the kernel invocation to the runtime.  
 * The kernel function is renamed and passed as one of the 
 * kernel invocation parameters.  
 */
public class ClearCUDASpecs
{
  private static String pass_name = "[ClearCUDASpecs-MCUDA]";
  private Program program;

  public ClearCUDASpecs(Program program)
  {
    this.program = program;
  }

  public static void run(Program program)
  {
    Tools.printlnStatus(pass_name + " begin", 1);

    ClearCUDASpecs pass = new ClearCUDASpecs(program);
    pass.start();

    Tools.printlnStatus(pass_name + " end", 1);
  }


  private void start()
  {
    DepthFirstIterator i = new DepthFirstIterator(program);
    i.pruneOn(VariableDeclaration.class);

    Set<Class<? extends Traversable>> set = new HashSet<Class<? extends Traversable>>();
    set.add(Procedure.class);
    set.add(VariableDeclaration.class);

    for (;;)
    {
      Procedure proc = null;
      VariableDeclaration decl = null;

      try {
        Object o = i.next(set);
        if (o instanceof Procedure)
          proc = (Procedure)o;
        else
          decl = (VariableDeclaration)o;
      } catch (NoSuchElementException e) {
        break;
      }

      if (proc != null)
      {
        Tools.printlnStatus(pass_name + " examining procedure " + proc.getName(), 2);

        // **AP** Remove SHARED specifier from parameter declarations
        List<Declaration> paramLst = proc.getParameters();
        for(Declaration paramDecl : paramLst) {
          List<Specifier> pSpecs = ((VariableDeclaration) paramDecl).getSpecifiers(); 
          while(pSpecs.remove(Specifier.SHARED));
          while(pSpecs.remove(Specifier.CONSTANT));
        }

        // Remove Procedure's specifiers
        proc.removeProcedureSpec(Specifier.GLOBAL);
        proc.removeProcedureSpec(Specifier.HOST);
        proc.removeProcedureSpec(Specifier.DEVICE);

      }
      else
      {
        while (decl.getSpecifiers().remove(Specifier.SHARED));
        while (decl.getSpecifiers().remove(Specifier.LOCAL));
        while (decl.getSpecifiers().remove(Specifier.DEVICE));
        while (decl.getSpecifiers().remove(Specifier.CONSTANT));
      }
    }

    i = new DepthFirstIterator(program);
    i.pruneOn(Procedure.class);

    //Now we need to do a separate pass for prototypes
    for (;;)
    {
      ProcedureDeclarator pDecl = null;

      try {
        //*cetus-1.1*   pDecl = i.next(ProcedureDeclarator.class);
        pDecl = (ProcedureDeclarator)i.next(ProcedureDeclarator.class);
      } catch (NoSuchElementException e) {
        break;
      }

      //Function prototypes are the only thing that will show up here, 
      //(Pruned on Procedure.)  The parent will have the leading specs, 
      //and is a variable type.
      VariableDeclaration prototype = 
        (VariableDeclaration)pDecl.getParent();
      //String functionName = pDecl.getDirectDeclarator().toString();

      List<Specifier> specs = prototype.getSpecifiers();
      while(specs.remove(Specifier.GLOBAL));
      while(specs.remove(Specifier.DEVICE));
      while(specs.remove(Specifier.HOST));


    }

  }
}
