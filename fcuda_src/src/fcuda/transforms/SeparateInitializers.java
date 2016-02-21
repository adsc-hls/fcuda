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

import fcuda.*;
import fcuda.utils.*;

import cetus.hir.*;
import cetus.exec.*;

/**
 * Transforms a program such that every variable declaration 
 * within a program has no initializer.  The declared initializers 
 * are separated into independent statements.  Requires that 
 * the single declaration pass has been run first.
 */
public class SeparateInitializers extends KernelTransformPass
{
  public SeparateInitializers(Program program)
  {
    super(program);
  }

  public void splitInitialization(Procedure proc)
  {
    DepthFirstIterator iter = new DepthFirstIterator(proc);

    while(iter.hasNext()) {
      DeclarationStatement declStmt;

      try {
        declStmt=(DeclarationStatement)iter.next(DeclarationStatement.class);
      } catch (NoSuchElementException e) {
        break;
      }

      //Skip over annotations, or other unknown declaration types.
      if(!(declStmt.getDeclaration() instanceof VariableDeclaration))
        continue;

      //TODO: make this compatible with function prototype 
      //declarations.
      VariableDeclaration vDecl = 
        (VariableDeclaration)declStmt.getDeclaration();

      VariableDeclarator variable = 
        (VariableDeclarator)vDecl.getDeclarator(0);

      Initializer vInit = variable.getInitializer();

      variable.setInitializer(null);

      if(vInit == null)
        continue;

      Tools.println("Separating initializer from variable " + variable.toString(), 3);

      // TODO: Check to see if the initialization is an invariant 
      // initialization of a const value
      vDecl.getSpecifiers().remove(Specifier.CONST);
      // Technically impossible?
      variable.getSpecifiers().remove(Specifier.CONST);


      //TODO: make compatible with array initializers.  
      //Only handles scalar expression initializers right now.
      List<Traversable> children = vInit.getChildren();

      if(children.size() > 1 ||
          !( children.get(0) instanceof Expression ) )
        throw new UnsupportedInput("Only primitive initializers are currently supported");


      Expression initExpr = 
        (Expression)((Expression)children.get(0)).clone();

     IDExpression vID = (IDExpression)variable.getID().clone();

      AssignmentExpression vAssign =
        new AssignmentExpression(vID, AssignmentOperator.NORMAL, initExpr);

      ExpressionStatement vStmt = new ExpressionStatement(vAssign);
      //Making some assumptions about the parse tree.
      //Parent of a Vdecl with an initializer must be a DeclStmt, and 
      //the parent of a DeclStmt must be a CompoundStmt

      CompoundStatement scope = (CompoundStatement)declStmt.getParent();

      scope.addStatementAfter(declStmt, vStmt);

      // **AP** Check for FCUDA annotated pragmas and move them to the newly created statement
      // **AP** Maybe it should also be done with other annotation types ...?
      if(declStmt.getAnnotations(FcudaAnnotation.class).size() != 0) {
        List<FcudaAnnotation> fcudAnnots = declStmt.getAnnotations(FcudaAnnotation.class);
        for (FcudaAnnotation fcAnnot : fcudAnnots) 
          vStmt.annotate(fcAnnot);

        declStmt.removeAnnotations(FcudaAnnotation.class);
      }

      }
    }

    public void transformProcedure(Procedure proc)
    {
      if (Driver.getOptionValue("Fcuda") != null) {
        List<Procedure> tskLst = FCUDAutils.getTaskMapping(proc.getSymbolName()); 
        if(tskLst != null) {
          for( Procedure task : tskLst )
            splitInitialization(task);
        } else 
          splitInitialization(proc);

      } else 
        splitInitialization(proc);
    }

    public String getPassName()
    {
      return new String("[SeparateInitializers]");
    }
  }
