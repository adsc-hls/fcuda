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
import fcuda.*;

import cetus.hir.*;

/* Utility class for keeping track of file-scoped declarations 
 * for transformed kernels */
public class GlobalKernelState
{
  public ClassDeclaration SM_state;
  public ClassDeclaration threadState;
  public Expression SM_struct_ref;
  //*cetus-1.1*  public Identifier kernelVars;
  public IDExpression kernelVars;
  //*cetus-1.1*   public Identifier localVars;
  public IDExpression localVars;
  //*cetus-1.1*  public Identifier SM_data_var;
  public IDExpression SM_data_var;
  //*cetus-1.1*   public Identifier lvStructID;
  public IDExpression lvStructID;

  public GlobalKernelState() {};

}




