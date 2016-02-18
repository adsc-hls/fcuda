package fcuda.ir;
import fcuda.*;

import cetus.hir.*;

import java.util.*;

public class ImplicitVariable 
{
  protected String name;
  protected List<Specifier> type;

  public ImplicitVariable(String s, Specifier varType)
  {
    name = s;
    type = (new ChainedList<Specifier>()).addLink(varType);
  }

  public ImplicitVariable(String s, List<Specifier> varType)
  {
    name = s;
    type = varType;
  }

  public String getString()
  {
    return new String(name);
  }

  public List<Expression> getId()
  {    
    return (new ChainedList<Expression>()).addLink(new NameID(getString()));
  }

  public List<VariableDeclaration> getDecl()
  {
    ChainedList<VariableDeclaration> retval = new ChainedList<VariableDeclaration>();
    return retval.addLink(new VariableDeclaration(type, 
           new VariableDeclarator((IDExpression)getId().get(0))));
  }

}
