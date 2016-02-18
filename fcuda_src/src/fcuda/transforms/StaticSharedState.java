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
