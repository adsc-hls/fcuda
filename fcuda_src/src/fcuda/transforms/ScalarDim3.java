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

import java.util.*;

import fcuda.ir.*;
import fcuda.utils.*;

import cetus.hir.*;
import cetus.exec.*;

/**
 * John A. Stratton
 * This class translates accesses to the implicitly defined 
 * dim3 variables of a CUDA program into scalar varible accesses.  
 *
 * This is currently required for Xpilot bridge support.
 */
public class ScalarDim3 extends KernelTransformPass
{
  private static List<Dim3Var> implicitvars;

  static {
    Class<?>[] params = new Class<?>[1];
    params[0] = int.class;
    implicitvars = new LinkedList<Dim3Var>();
    //Add each implicit variable, and the method which will 
    // produce the correct representation of it for the options 
    // currently set.
    implicitvars.add(MCUDAUtils.Bidx);
    implicitvars.add(MCUDAUtils.Tidx);
    implicitvars.add(MCUDAUtils.Bdim);
    implicitvars.add(MCUDAUtils.Gdim);
    implicitvars.add(MCUDAUtils.StartIdx);
    implicitvars.add(MCUDAUtils.EndIdx);
  }

  public String getPassName()
  {
    return new String("[ScalarDim3-MCUDA]");
  }

  public ScalarDim3(Program program)
  {
    super(program);
  }

  public void transformProcedure(Procedure proc)
  {
    //For each variable and field, look for that variable-field access
    //  combination, and replace it with the correct representation.
    for(Dim3Var var : implicitvars)
    {
      for(int entry = 0; entry < var.getNumEntries(); entry++)
      {
        AccessExpression old = 
          //*cetus-1.1*  new AccessExpression(new Identifier(var.getString()), 
          new AccessExpression(new NameID(var.getString()), 
              AccessOperator.MEMBER_ACCESS,
              //*cetus-1.1*  new Identifier(var.getDimEntry(entry)));
              new NameID(var.getDimEntry(entry)));

        Expression replacement = var.getId(entry);

        if(Driver.getOptionValue("CEAN") != null && var == MCUDAUtils.Tidx)
          if(Driver.getOptionValue("CEANv2") != null) 
            for(Expression e : MCUDAUtils.getBdim())
              replacement = new ArrayAccess(replacement, new ArraySlice());

          else
            replacement = new ArrayAccess(replacement, new ArraySlice());

        Tools.replaceAll(proc, old, replacement);
      }
    }

  }

}
