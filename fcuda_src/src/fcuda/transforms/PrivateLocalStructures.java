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

public class PrivateLocalStructures extends LocalStateTransform
{
  private GlobalKernelState gks;

  public PrivateLocalStructures(GlobalKernelState gks)
  {
    this.gks = gks;
  }

  @SuppressWarnings("unused")
  protected void TransformDecl(VariableDeclaration vDecl, Traversable root)
  {
    VariableDeclarator declarator = 
      (VariableDeclarator)vDecl.getDeclarator(0);
    List<Specifier> specs = declarator.getArraySpecifiers();

    specs.add(0, new ArraySpecifier(MCUDAUtils.getMaxNumThreads()));
  }

  protected Expression getAccessExpr(VariableDeclaration vDecl)
  {
    //*cetus-1.1*  IDExpression symbol = vDecl.getDeclarator(0).getSymbol();
    IDExpression symbol = vDecl.getDeclarator(0).getID();
    //Make a template array access to copy from
    ArrayAccess variableStore = 
      new ArrayAccess((Expression)symbol.clone(), MCUDAUtils.LocalTidx.getId().get(0));

    variableStore.setParens(true);

    return variableStore;
  }

  protected Expression getZeroAccessExpr(VariableDeclaration vDecl)
  {
    //*cetus-1.1*  IDExpression symbol = vDecl.getDeclarator(0).getSymbol();
    IDExpression symbol = vDecl.getDeclarator(0).getID();
    //Make a template array access to copy from
    ArrayAccess variableStoreZero = 
      new ArrayAccess((Expression)symbol.clone(), new IntegerLiteral(0));

    variableStoreZero.setParens(true);

    return variableStoreZero;
  }

  @SuppressWarnings("unused")
  protected void finalize(Procedure proc) {};

  @SuppressWarnings("unused")
  public void extraFunc(List<VariableDeclaration> vDecls, Procedure proc)
  {
    for(VariableDeclaration vDecl : vDecls)
    {
      Traversable root = vDecl.getParent().getParent();
      //And for every instance referencing this declaration, 
      List<Expression> uses = MCUDAUtils.getInstances(vDecl, root);
      //First, make the declarator itself an array
      TransformDecl(vDecl, root);

      VariableDeclarator declarator = 
        (VariableDeclarator)vDecl.getDeclarator(0);
      //Make a template array access to copy from
      ArrayAccess variableStore = 
        //*cetus-1.1*    new ArrayAccess((Expression)declarator.getDirectDeclarator().clone(),
        new ArrayAccess((Expression)declarator.getID().clone(),
            MCUDAUtils.LocalTidx.getId().get(0));

      ArrayAccess variableStoreZero = 
        //*cetus-1.1*   new ArrayAccess((Expression)declarator.getDirectDeclarator().clone(),
        new ArrayAccess((Expression)declarator.getID().clone(),
            new IntegerLiteral(0));
      variableStore.setParens(true);
      variableStoreZero.setParens(true);


      //make it an array access.
      Tools.printStatus("transforming decl: " + uses.size() + " uses", 4);
      MCUDAUtils.swapWithExprs(uses, variableStore, variableStoreZero);
    }
  }

}
