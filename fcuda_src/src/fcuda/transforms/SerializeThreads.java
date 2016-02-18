package fcuda.transforms;

import java.util.*;

import fcuda.utils.*;
import fcuda.common.*;

import cetus.hir.*;
import cetus.exec.*;

/**
 * Interfaces a function call to the MCUDA runtime library
 * First creates a type definition for this function's parameter
 * struction
 * Then modifies the current function to now accept the implicit
 * block and grid variables of the CUDA programming model,
 * in addition to a pointer to an instance of the parameter structure
 * Lastly, creates a new function with the original function
 * signature, whose sole purpose is to construct the paramter
 * structure, and pass the kernel invocation to the runtime
 * The kernel function is renamed and passed as one of the
 * kernel invocation parameters.
 */

public class SerializeThreads extends KernelTransformPass
{
  public String getPassName()
  {
    return new String("[SerializeThreads-MCUDA]");
  }

  public SerializeThreads(Program program)
  {
    super(program);
  }

  private Procedure mProcedure;

  // *AP* Created new function in order to be able to handle
  //      FCUDA task procedures
  public void serialize(Procedure proc)
  {
    CompoundStatement body = proc.getBody();
    CompoundStatement newBody = new CompoundStatement();

    proc.setBody(newBody);

    List<VariableDeclaration> tidDecl = MCUDAUtils.Tidx.getDecl();
    //Slightly cheating here.  We're going to call the
    //thread indices "shared" variables, to eliminate
    //later issues with accessing non-shared variables
    //using the thread index

    for (VariableDeclaration currentDecl : tidDecl) {
      currentDecl.getSpecifiers().add(Specifier.SHARED);
      newBody.addDeclaration(currentDecl);
    }
    int numDims = FCUDAGlobalData.getKernTblkDim(mProcedure);
    newBody.addStatement(MCUDAUtils.NewNestedThreadLoop(numDims, body));
  }

  public void transformProcedure(Procedure proc)
  {
    mProcedure = proc;
    List<Procedure> tskLst = FCUDAutils.getTaskMapping(proc.getSymbolName());
    //Tools.printlnStatus("Task Procs: "+tskLst.toString(), 0);
    if (tskLst != null) {
      for (Procedure task : tskLst)
        if (FCUDAutils.getTaskType(task).equals("compute"))
          serialize(task);
    }
  }
}
