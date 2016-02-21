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

import java.util.*;

import cetus.exec.*;
import cetus.hir.*;
import cetus.transforms.*;

public abstract class KernelTransformPass extends TransformPass
{
  protected KernelTransformPass(Program program)
  {
    super(program);
  }

  public abstract void transformProcedure(Procedure proc);

  //Override this function if something needs to be applied to 
  //kernel prototypes as well.
  @SuppressWarnings("unused")
  public void transformPrototype(ProcedureDeclarator pDecl)
  {
    return;
  }

  public void start()
  {
    HashSet<String> skip_set = Driver.getSkipProcedureSet();

    BreadthFirstIterator iter = new BreadthFirstIterator(program);
    iter.pruneOn(Procedure.class);
    Set<Class<? extends Traversable>> set = new HashSet<Class<? extends Traversable>>();
    set.add(Procedure.class);
    set.add(ProcedureDeclarator.class);

    for (;;) {
      Procedure proc = null;
      ProcedureDeclarator pDecl = null;

      try {
        Object o = iter.next(set);
        if(o instanceof Procedure)
          proc = (Procedure)o;
        else //ProcedureDeclarator
          pDecl = (ProcedureDeclarator)o;
      } catch (NoSuchElementException e) {
        break;
      }

      if(proc != null) {
        List<Specifier> specs = proc.getReturnType();
        if(!(specs.contains(Specifier.GLOBAL)))
          continue;

        if (!skip_set.contains(proc.getName().toString())) {
          System.out.println(getPassName() + " examining procedure " + proc.getName());
          transformProcedure(proc);
        }
        else {
          System.out.println(getPassName() + " skipping procedure " + proc.getName());
        }
      }
      else if (pDecl != null) {
        //Function prototypes are the only thing that will show up here, 
        //(Pruned on Procedure.)  The parent will have the leading specs, 
        //and is a variable type.
        VariableDeclaration prototype = 
          (VariableDeclaration)pDecl.getParent();

        String functionName = pDecl.getID().toString();

        List<Specifier> specs = prototype.getSpecifiers();

        if(!(specs.contains(Specifier.GLOBAL)))
          continue;

        if (!skip_set.contains(functionName)) {
          System.out.println(getPassName() + " examining function " + functionName);
          transformPrototype(pDecl);
        }
        else {
          System.out.println(getPassName() + " skipping function " + functionName);
        }
      }

    }
  }
}
