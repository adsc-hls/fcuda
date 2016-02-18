package fcuda.ir;

import fcuda.utils.*;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;

import cetus.hir.*;
import cetus.exec.*;

public class ThreadLoop extends Statement implements Loop
{
  private static Method class_print_method;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = ThreadLoop.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  public ThreadLoop(Statement body, int numThrBlkDims)
  {
    super(1);
    
    object_print_method = class_print_method;

    setNumDims(numThrBlkDims);

    children.add(null);  // init1         -- x dim
    setInit(0, (Expression) new IntegerLiteral(0) );
    children.add(null);  // bound1        -- x dim
    setBound(0, MCUDAUtils.getBdimID(0));
    children.add(null);  // update1       -- x dim
    setUpdate(0, (Expression) new BinaryExpression(MCUDAUtils.getTidID(0), BinaryOperator.ADD, new IntegerLiteral(1)));
    if (numThrBlkDims - 1 > 0) {
      children.add(null);  // init2         -- y dim
      setInit(1, (Expression) new IntegerLiteral(0) );
      children.add(null);	 // bound2        -- y dim
      setBound(1, MCUDAUtils.getBdimID(1));
      children.add(null);	 // update2       -- y dim
      setUpdate(1, (Expression) new BinaryExpression(MCUDAUtils.getTidID(1), BinaryOperator.ADD, new IntegerLiteral(1)));
    }
    if (numThrBlkDims - 2 > 0) {
      children.add(null);  // init3         -- z dim
      setInit(2, (Expression) new IntegerLiteral(0) );
      children.add(null);	 // bound3        -- z dim
      setBound(2, MCUDAUtils.getBdimID(2));
      children.add(null);	 // update3       -- z dim
      setUpdate(2, (Expression) new BinaryExpression(MCUDAUtils.getTidID(2), BinaryOperator.ADD, new IntegerLiteral(1)));
    }
    children.add(new CompoundStatement());

    setBody(body);
  }

  private int numDims;

  public void setNumDims(int numThrBlkDims)
  {
    numDims = numThrBlkDims;
  }

  public int getNumDims()
  {
    return numDims;
  }

  /* SymbolTable interface */
  public void addDeclaration(Declaration decl)
  {
    ((CompoundStatement)getBody()).addDeclaration(decl);
  }

  /* SymbolTable interface */
  public void addDeclarationBefore(Declaration ref, Declaration decl)
  {
    ((CompoundStatement)getBody()).addDeclarationBefore(ref,decl);
  }

  /* SymbolTable interface */
  public void addDeclarationAfter(Declaration ref, Declaration decl)
  {
    ((CompoundStatement)getBody()).addDeclarationAfter(ref,decl);
  }

  public Expression getInit(int dimId)
  {
    int childId = dimId*3;
    if (childId >= 3 * numDims) {
      return null;
    }
    return (Expression)children.get(childId);
  }

  public Expression getBound(int dimId)
  {
    int childId = dimId*3+1;
    if (childId >= 3 * numDims+1)
      return null;
    return (Expression)children.get(childId);
  }

  public Expression getUpdate(int dimId)
  {
    int childId = dimId*3+2;
    if (childId >= 3 * numDims+2)
      return null;
    return (Expression)children.get(childId);
  }

  public void setInit(int dimId, Expression init) 
  {
    int childId = dimId*3;
    if (getInit(dimId) != null)
      getInit(dimId).setParent(null);
  
    children.set(childId, init);

    if (init != null) {
      init.setParent(this);
      init.setParens(false);
    }
  }

  public void setBound(int dimId, Expression bound) 
  {
    int childId = dimId*3+1;
    if (getBound(dimId) != null)
      getBound(dimId).setParent(null);

    children.set(childId, bound);

    if (bound != null) {
      bound.setParent(this);
      bound.setParens(false);
    }
  }

  public void setUpdate(int dimId, Expression update) 
  {
    int childId = dimId*3+2;
    if (getUpdate(dimId) != null)
      getUpdate(dimId).setParent(null);

    children.set(childId, update);

    if (update != null) {
      update.setParent(this);
      update.setParens(false);
    }
  }



  /**
   * Prints a loop to a stream.
   *
   * @param loop The loop to print.
   * @param stream The stream on which to print the loop.
   */ 
  public static void defaultPrint(ThreadLoop loop, PrintWriter pw)
  {
    int numDims = loop.getNumDims();
    String ltid = MCUDAUtils.LocalTidx.getString();
    for (int i = 3; i > 0; i--) {
      // Don't print loops with no initialization
      // This is currently the means for compiler passes to disable some or all iteration dimensions
      if (loop.getInit((i-1)) == null)
        continue;
      String tid = MCUDAUtils.Tidx.getString(i-1);
      pw.print("for ("+tid+"="+loop.getInit(i-1).toString());
      if (i == numDims && Driver.getOptionValue("Fcuda") == null)  
        pw.print(", "+ltid+"=0");
      pw.print(";"+tid+"<"+loop.getBound(i-1).toString()+" ; "+tid+"="+loop.getUpdate(i-1).toString());
      if (i == 1  && Driver.getOptionValue("Fcuda") == null)
        pw.print(", "+ltid+"++)");
      else
        pw.print(") \n");
    }

    loop.getBody().print(pw);
  }
  

  // *AP* Re-writing this toString function to take into consideration the programmable  
  // *AP* init, bound and update expressions. These are used in Fcuda to implement 
  // *AP* unrolling and memory partitioning. These changes can be copied to the other
  // *AP* toString() functions if required.
  // ***********************************************************************************
  public String toString_Nested()
  {
    StringBuilder str = new StringBuilder(20);
    String ltid = MCUDAUtils.LocalTidx.getString();

    for (int i = 3; i > 0; i--) {
      // Don't print loops with no initialization
      // This is currently the means for compiler passes to disable some or all iteration dimensions
      if (getInit((i-1)) == null)
        continue;
      String tid = MCUDAUtils.Tidx.getString(i-1);
      String bdim = MCUDAUtils.Bdim.getString(i-1);
      str.append("for ("+tid+"="+getInit(i-1).toString());
      // **AP** Currently do not need local threadId in Fcuda
      if (i == numDims && Driver.getOptionValue("Fcuda") == null)  
        str.append(", "+ltid+"=0");
      str.append(";"+tid+"<"+getBound(i-1).toString()+" ; "+tid+"="+getUpdate(i-1).toString());
      if (i == 1 && Driver.getOptionValue("Fcuda") == null)
        str.append(", "+ltid+"++)\n");
      else
        str.append(")\n");
    }
    str.append(getBody().toString());
    return str.toString();
  }

  public String toString_Range()
  {
    StringBuilder str = new StringBuilder(20);
    String numDimString = Driver.getOptionValue("numDims");
    int numDims = 3;
    if (numDimString != null)
      numDims = Integer.parseInt(numDimString);
    String ltid = MCUDAUtils.LocalTidx.getString();

    for (int i = numDims; i > 0; i--)
    {
      String tid = MCUDAUtils.Tidx.getString(i-1);
      String start = MCUDAUtils.StartIdx.getString(i-1);
      String end = MCUDAUtils.EndIdx.getString(i-1);

      str.append("for ("+tid+"="+start);
      if (i == numDims)
        str.append(", "+ltid+"=0");
      str.append(";"+tid+"<"+end+" ; "+tid+"++");
      if (i == 1)
        str.append(", "+ltid+"++)\n");
      else
        str.append(") {\n");
    }
    str.append(getBody().toString());
    for (int i = numDims; i > 1; i--)
      str.append("}\n");
    return str.toString();
  }


  public String toString_AtomicLib()
  {
    StringBuilder str = new StringBuilder(20);
    String ltid = MCUDAUtils.LocalTidx.getString();
    String numDims = Driver.getOptionValue("numDims");
    
    str.append("while( ("+ltid+" = atomic_get_next_tid_");
    if (numDims == null || numDims.compareTo("2") < 0)
      str.append("1d(&"+MCUDAUtils.Tidx.getString());
    else {
      if (numDims.compareTo("2") > 0) {
        str.append("3d(&"+MCUDAUtils.Tidx.getString()+",");
        str.append("&"+MCUDAUtils.Bdim.getString());
      } else {
        str.append("2d(&"+MCUDAUtils.Tidx.getString()+",");
        str.append(MCUDAUtils.Bdim.getString(0));
      }
    }
    str.append(") ) >= 0 )\n");
    str.append(getBody().toString());
    str.append("\n");
    return str.toString();
  }

  public String toString()
  {
    if (Driver.getOptionValue("dynamicTasks") != null)
      return toString_AtomicLib();
    else if (Driver.getOptionValue("staticTasks") != null)
      return toString_Range();
    else 
      return toString_Nested();
  }

  public Statement getBody()
  {
    int index = numDims * 3;
    return (Statement)children.get(index);
  }

  public Expression getCondition()
  {
    return null;
  }

  public void setCondition(Expression cond)
  {
    throw new InternalError();
  }

  /* SymbolTable interface */
  public List<SymbolTable> getParentTables()
  {
    List<SymbolTable> ret = new ArrayList<SymbolTable>();
    Traversable p = getParent();
    while (p != null) {
      if (p instanceof SymbolTable) {
        ret.add((SymbolTable)p);
      }
      p = p.getParent();
    }
    return ret;
  }

  public void setBody(Statement body)
  {
    if (getBody() != null)
      getBody().setParent(null);

    if (body == null) {
      body = new CompoundStatement();
    }
    else if (!(body instanceof CompoundStatement)) {
      CompoundStatement cs = new CompoundStatement();
      cs.addStatement(body);
      body = cs;
    }
    children.set(numDims * 3, body);
    body.setParent(this);
  }

  /**
   * Overrides the class print method, so that all subsequently
   * created objects will use the supplied method.
   *
   * @param m The new print method.
   */
  static public void setClassPrintMethod(Method m)
  {
    class_print_method = m;
  }

}
