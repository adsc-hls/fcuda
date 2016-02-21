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

import java.util.*;

import cetus.hir.*;

public class StaticSharedState implements BlockStateTransform 
{
  private final GlobalKernelState gks;

  public StaticSharedState(GlobalKernelState gks)
  { 
    this.gks = gks;
  }

  public void TransformSharedVars(List<VariableDeclaration> vDecls)
  {
    for(VariableDeclaration vDecl : vDecls)
    {
      Traversable root = vDecl.getParent().getParent();
      List<Expression> refs = MCUDAUtils.getInstances(vDecl, root);
      //*cetus-1.1*   IDExpression varID = vDecl.getDeclarator(0).getSymbol();
      IDExpression varID = vDecl.getDeclarator(0).getID();
      ((Statement)vDecl.getParent()).detach();
      gks.SM_state.addDeclaration(vDecl); //Promote to structure
      AccessExpression svar_access = 
        new AccessExpression(gks.SM_data_var, AccessOperator.POINTER_ACCESS, varID);
      MCUDAUtils.swapWithExprs(refs, svar_access, svar_access);
    }
  }
}   
