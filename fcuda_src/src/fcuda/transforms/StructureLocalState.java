package fcuda.transforms;  
import fcuda.utils.*;

import java.util.*;

import cetus.hir.*;
import cetus.exec.*;

public class StructureLocalState extends LocalStateTransform 
{
  protected final GlobalKernelState gks;

  public StructureLocalState(GlobalKernelState gks)
  { 
    this.gks = gks;
  }

  protected void TransformDecl(VariableDeclaration vDecl, Traversable root)
  {
    //Remove Decl from current position, and place in structure in array form
    //Relies on Single Declarator property for correctness
    root.removeChild(vDecl.getParent());
    gks.threadState.addDeclaration(vDecl);
  }

  private Expression getAccess(VariableDeclaration vDecl, Expression index)
  {
    //*cetus-1.1*  Expression symbol = vDecl.getDeclarator(0).getSymbol();
    Expression symbol = vDecl.getDeclarator(0).getID();
    Expression structure = (IDExpression)gks.localVars.clone();

    if(Driver.getOptionValue("staticLocalState") != null)
      structure = 
        new AccessExpression((Expression)gks.SM_data_var.clone(), 
            AccessOperator.POINTER_ACCESS, (IDExpression) structure);

    ArrayAccess threadVars = new ArrayAccess(structure, index);

    AccessExpression var_access = 
      new AccessExpression(threadVars, AccessOperator.MEMBER_ACCESS, 
          (IDExpression)symbol.clone());

    var_access.setParens(true);

    return var_access;
  }    

  protected Expression getAccessExpr(VariableDeclaration vDecl)
  {
    return getAccess(vDecl, MCUDAUtils.LocalTidx.getId().get(0));
  }

  protected Expression getZeroAccessExpr(VariableDeclaration vDecl)
  {
    return getAccess(vDecl, new IntegerLiteral(0));
  }


  protected void finalize(Procedure proc)
  {
    //Add the class declaration before the procedure
    TranslationUnit tu = (TranslationUnit)proc.getParent();
    tu.addDeclarationBefore(proc, gks.threadState);
    MCUDAUtils.addStructTypedefBefore(tu, proc, gks.threadState);

    //Add local vars structure array to class
    VariableDeclarator lv_declarator = new VariableDeclarator( gks.localVars, 
        new ArraySpecifier(MCUDAUtils.getMaxNumThreads() )  );

    VariableDeclaration lv_decl = 
      new VariableDeclaration(
          new UserSpecifier((Identifier)gks.lvStructID.clone()), lv_declarator);

    if(Driver.getOptionValue("staticLocalState") != null)
      gks.SM_state.addDeclaration(lv_decl);
    else
      proc.getBody().addDeclaration(lv_decl);

  }

}   
