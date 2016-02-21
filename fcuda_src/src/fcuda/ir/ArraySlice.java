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

package fcuda.ir;

import cetus.hir.*;

import java.util.*;
import java.lang.reflect.*;
import java.io.*;

/**
 * An expression that defines an array slice.
 * Although Cetus does not enforce this constraint, this Expression 
 * type should only ever be a child of an ArrayAccess expression.
 */
public class ArraySlice extends Expression
{
  private static Method class_print_method;

  static
  {
    Class<?>[] params = new Class<?>[2];

    try {
      params[0] = ArraySlice.class;
      //*cetus-1.1*   params[1] = OutputStream.class;
      params[1] = PrintWriter.class;
      class_print_method = params[0].getMethod("defaultPrint", params);
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
  }

  public ArraySlice()
  {
    super();
    object_print_method = class_print_method;
  }

  public ArraySlice(Expression begin, Expression span)
  {
    super();
    children.add(begin);
    children.add(span);
    object_print_method = class_print_method;
  }

  public static void defaultPrint(ArraySlice as, PrintWriter pw)
  {
    if(as.children.size() != 0)
      as.children.get(0).print(pw);

    pw.print(":");

    if(as.children.size() >= 2)
      as.children.get(0).print(pw);
  }

  public String toString()
  {
    StringBuilder str = new StringBuilder(10);
    if(children.size() != 0)
      str.append(children.get(0).toString());
    str.append(':');
    if(children.size() >= 2)
      str.append(children.get(1).toString());
    return str.toString();
  }
}
