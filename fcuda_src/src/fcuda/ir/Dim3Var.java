package fcuda.ir;
import fcuda.*;

import cetus.hir.*;
import cetus.exec.*;

import java.util.*;

public class Dim3Var extends ImplicitVariable 
{
  private int numEntries;

  private static ArrayList<String> dimEntries;

  static {
    dimEntries = new ArrayList<String>();
    dimEntries.add(new String("x"));
    dimEntries.add(new String("y"));
    dimEntries.add(new String("z"));
  }

  private static String getDim3String()
  {
    return new String("dim3");
  }

  private static IDExpression getDim3ID()
  {
    return new NameID(getDim3String());
  }

  private static UserSpecifier getDim3Type()
  {
    return new UserSpecifier(getDim3ID());
  }

  public int getNumEntries()
  {
    return numEntries;
  }

  public String getDimEntry(int idx)
  {
    if(idx < numEntries)
      return new String(dimEntries.get(idx));
    else 
      throw new UnsupportedOperationException(numEntries+"-D dimensions only");
  }

  public Dim3Var(String s, int numDims)
  {
    super(s, getDim3Type());
    numEntries = numDims;
  }

  public String getString(int dim)
  {
    char access = '.';
    if(Driver.getOptionValue("scalarDim3") != null)
      access = '_';    
    return name + access + getDimEntry(dim);
  }

  public List<Expression> getId()
  {    
    if(Driver.getOptionValue("scalarDim3") == null)
    {
      return super.getId();
    }
    else {
      List<Expression> retval = new ChainedList<Expression>();
      //Non-recursive
      for(int i = 0; i < numEntries; i++)
        retval.add( getId(i) );
      return retval;
    }
  }

  public List<Expression> getHostId()
  {    
    return super.getId();
  }

  public List<Expression> getInterfaceId()
  {    
    if(Driver.getOptionValue("scalarDim3") == null)
      return super.getId();
    else {
      List<Expression> retval = new ChainedList<Expression>();
      //Non-recursive
      for(int i = 0; i < numEntries; i++)
        retval.add( getHostId(i) );
      return retval;
    }
  }

  public Expression getId(int entry)
  {
    if(Driver.getOptionValue("scalarDim3") == null) {
      return new AccessExpression(getId().get(0), 
          AccessOperator.MEMBER_ACCESS,
             new NameID(getDimEntry(entry)));
    }
    else {
      return new NameID(getString(entry));
    }
  }

  public Expression getHostId(int entry)
  {
    return new AccessExpression(getHostId().get(0), 
        AccessOperator.MEMBER_ACCESS,
           new NameID(getDimEntry(entry)));
  }

  public List<VariableDeclaration> getDecl()
  {
    if(Driver.getOptionValue("scalarDim3") == null)
    {
      return super.getDecl();
    }
    else {
      List<VariableDeclaration> retval = new ChainedList<VariableDeclaration>();
      for(int i = 0; i < numEntries; i++)
        retval.add(new VariableDeclaration(Specifier.INT,
              new VariableDeclarator((Identifier)getId(i))));
      return retval;
    }
  }

  public List<VariableDeclaration> getHostDecl()
  {
    ChainedList<VariableDeclaration> retval = new ChainedList<VariableDeclaration>();
    return retval.addLink(new VariableDeclaration(type, 
          new VariableDeclarator((Identifier)getHostId().get(0))));
  }
}
