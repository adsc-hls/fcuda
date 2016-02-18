package fcuda.transforms;
import fcuda.utils.*;

import java.util.*;
import cetus.hir.*;

/* Utility interface for abstracting away the transformations applied 
 * to thread-private kernel state */
public abstract class LocalStateTransform
{
  protected abstract void TransformDecl(VariableDeclaration vDecl, Traversable root);

  protected abstract Expression getAccessExpr(VariableDeclaration vDecl);

  protected abstract Expression getZeroAccessExpr(VariableDeclaration vDecl);

  protected abstract void finalize(Procedure proc);

  public void TransformLocals(List<VariableDeclaration> vDecls, 
      Procedure proc) 
  {
    for(VariableDeclaration vDecl : vDecls)
    {
      Traversable root = vDecl.getParent().getParent();
      List<Expression> uses = MCUDAUtils.getInstances(vDecl, root);

      Expression var_access = getAccessExpr(vDecl);

      Expression varZero_access = getZeroAccessExpr(vDecl);

      Tools.printlnStatus("transforming decl " + vDecl.toString() + ": " + uses.size() + " uses", 4);

      TransformDecl(vDecl, root);

      MCUDAUtils.swapWithExprs(uses, var_access, varZero_access); 
    }

    finalize(proc);
  }

}

