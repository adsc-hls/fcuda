package fcuda.transforms;

import java.util.*;

import fcuda.ir.*;
import fcuda.utils.*;
import fcuda.*;

import cetus.hir.*;
import cetus.exec.*;

public class ArrayLocalState extends LocalStateTransform
{
  protected final GlobalKernelState gks;

  public ArrayLocalState(GlobalKernelState gks)
  {
    this.gks = gks;
  }

  @SuppressWarnings("unused")
  protected void TransformDecl(VariableDeclaration vDecl, Traversable root)
  {
    VariableDeclarator declarator = 
      (VariableDeclarator)vDecl.getDeclarator(0);
    List<Specifier> specs = declarator.getArraySpecifiers();

    if(Driver.getOptionValue("CEANv2") == null)
      specs.add(0, new ArraySpecifier(MCUDAUtils.getMaxNumThreads()));
    else 
    {
      List<Expression> bdim_size = MCUDAUtils.getBdim();
      for(int i = bdim_size.size() - 1; i >= 0; i--)
        specs.add(0, new ArraySpecifier(bdim_size.get(i)));
    }

    if(Driver.getOptionValue("staticLocalState") != null)
    {
      root.removeChild(vDecl.getParent());
      gks.SM_state.addDeclaration(vDecl);
    }
  }

  private Expression getAccess(VariableDeclaration vDecl, Expression index)
  {
    //*cetus_1.1*    Expression symbol = vDecl.getDeclarator(0).getSymbol();
    Expression symbol = vDecl.getDeclarator(0).getID();
    Expression result;

    if(Driver.getOptionValue("staticLocalState") != null)
      symbol = 
        new AccessExpression(gks.SM_data_var, AccessOperator.POINTER_ACCESS, 
            (IDExpression)symbol);

    if(Driver.getOptionValue("CEANv2") == null)
      result = new ArrayAccess((Expression)symbol.clone(), index);
    else {
      result = (Expression)symbol.clone();
      for(Expression e : MCUDAUtils.getBdim())
      {
        //The size of the returned list gives us the dimensionality of each 
        //  expanded local variable, even though we don't care about the actual 
        //  size of each dimension.
        result = new ArrayAccess(result, index);
      }
    }

    result.setParens(true);

    return result;
  }

  protected Expression getAccessExpr(VariableDeclaration vDecl)
  {
    return getAccess(vDecl, MCUDAUtils.LocalTidx.getId().get(0));
  }

  protected Expression getZeroAccessExpr(VariableDeclaration vDecl)
  {
    return getAccess(vDecl, new IntegerLiteral(0));
  }

  protected boolean isVectorizable(ThreadLoop tloop)
  {
    return true;
    /*
       DepthFirstIterator iter = new DepthFirstIterator(tloop);

       while(iter.hasNext())
       {
       */
       }

  protected void transform(ThreadLoop tloop)
  {
    Tools.replaceAll(tloop, MCUDAUtils.LocalTidx.getId().get(0), new ArraySlice());
    tloop.setInit(0, null);
    tloop.setInit(1, null);
    tloop.setInit(2, null);
    Expression as = new ArraySlice();
    if(Driver.getOptionValue("CEANv2") != null)
    {
      DepthFirstIterator iter = new DepthFirstIterator(tloop);
      iter.pruneOn(Expression.class);

      while(iter.hasNext())
      {
        AssignmentExpression e = null;
        //*cetus-1.1*  try{ e = iter.next(AssignmentExpression.class); }
        try{ e = (AssignmentExpression)iter.next(AssignmentExpression.class); }
        catch( NoSuchElementException err ) {  break;  }
        // We know it's a local variable if the number of ArrayAccess and ArraySlice 
        //  expressions are equal to the number of bdims
        DepthFirstIterator iter2 = new DepthFirstIterator(e.getLHS());
        int accessCount = 0;
        boolean allAccesses = true;

        while(iter2.hasNext())
        {
          ArrayAccess aa = null;
          //*cetus-1.1*    try{ aa = iter2.next(ArrayAccess.class); }
          try{ aa = (ArrayAccess)iter2.next(ArrayAccess.class); }
          catch( NoSuchElementException err ) { break; }
          //Assuming only one index per array access: currently true
          if( aa.getIndex(0) instanceof ArraySlice)
            accessCount++;
          else {
            allAccesses = false;
            break;
          }
        }

        if(allAccesses && (accessCount == MCUDAUtils.getBdim().size()) )
        {
          int localdim = 0;
          for(int i = 0; i < MCUDAUtils.Bdim.getNumEntries(); i++)
          {
            if(MCUDAUtils.getBdim(i) > 1)
            {
              Tools.println("Replacing " + MCUDAUtils.Tidx.getId().get(i).toString() +
                  " with " + "__sec_implicit_index(" + localdim + ")" + 
                  " in " + e.toString(), 2);
              Tools.replaceAll(e, MCUDAUtils.Tidx.getId().get(i), 
                  //*cetus-1.1*  new Identifier("__sec_implicit_index(" + localdim++ + ")"));
                new NameID("__sec_implicit_index(" + localdim++ + ")"));
            }
          }
        }
      }

      //For each assignment.  
      //  If LHS is local, replace Tid on RHS with implicit index.
      //  If LHS is shared and accessed with Tidx
      //    Replace Tid on RHS, translating Tid to target dim
      //    Replace Tid on LHS with slice
    }
  }

  public void promoteToCEAN(Procedure proc)
  {
    DepthFirstIterator iter = new DepthFirstIterator(proc);
    iter.pruneOn(ThreadLoop.class);

    //Insert new ThreadLoop to initialize threadIdx arrays

    while(iter.hasNext())
    {
      ThreadLoop tloop;

      try {
        //*cetus-1.1*   tloop = iter.next(ThreadLoop.class);
        tloop = (ThreadLoop)iter.next(ThreadLoop.class);
      } catch (NoSuchElementException e) {
        break;
      }

      if(isVectorizable(tloop))
        transform(tloop);
    }
  }

  @SuppressWarnings("unused")
  protected void finalize(Procedure proc) { 
    if(Driver.getOptionValue("CEAN") != null)
      promoteToCEAN(proc);
  };

  }
