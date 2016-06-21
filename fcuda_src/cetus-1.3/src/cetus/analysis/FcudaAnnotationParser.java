package cetus.analysis;

import java.util.*;
import cetus.hir.*;

/**
 * Parses string text within an Fcuda annotation of type #pragma FCUDA ...
 * The parsed version is indexed by possible keys and the output is stored in
 * a map
 */
public class FcudaAnnotationParser
{
  private static String [] token_array;
  private static int token_index;
  private	static HashMap<String, Object> fcuda_map;

  public FcudaAnnotationParser() {
  }

  private static String get_token()
  {
    return token_array[token_index++];
  }

  /* Match a token with the input istr, then move to the next token */
  private static boolean match(String istr)
  {
    boolean answer = check(istr);
    if (answer == false) {
      System.out.println("FcudaAnnotationParser Syntax Error");
      System.out.println(display_tokens());
    }
    token_index++;
    return answer;
  }

  /* Match a token with the given string, but don't comsume the token */
  private static boolean check(String istr)
  {
    if ( end_of_token() ) 
      return false;
    return ( token_array[token_index].compareTo(istr) == 0 ) ? true : false;
  }	


  private static String display_tokens()
  {
    StringBuilder str = new StringBuilder(500);

    for (int i=0; i<token_array.length; i++)
    {
      str.append("token_array[" + i + "] = " + token_array[i] + "\n");
    }
    return str.toString();
  }

  private static boolean end_of_token()
  {
    return (token_index >= token_array.length) ? true : false;
  }

  /*
   * Parses pragmas that can be used with the "FCUDA" keyword
   * @return
   */
  public static boolean parse_pragma(HashMap<String, Object> input_map, String [] str_array)
  {
    boolean attachable_pragma = false;

    fcuda_map = input_map;
    token_array = str_array;
    int i;
    for(i=0;i<str_array.length;++i)
      System.out.println("Token "+str_array[i]);
    /* Start from token 3 as #, pragma, FCUDA have already been matched */
    token_index = 3;

    Tools.println(display_tokens(), 9);

    while (end_of_token() == false)
    {
      String construct = "fcuda_" + get_token();
      switch (fcuda_pragma.valueOf(construct)) {
	case fcuda_GRID:
	  parse_fcuda_grid();
	  attachable_pragma = true;
	  break;
	case fcuda_BLOCKS:
	  parse_fcuda_blocks();
	  attachable_pragma = true;
	  break;
	case fcuda_SYNC:
	  parse_fcuda_sync();
	  attachable_pragma = true;
	  break;
	case fcuda_TRANSFER:
	  parse_fcuda_transfer();
	  attachable_pragma = false;
	  break;
	case fcuda_COMPUTE:
	  parse_fcuda_compute();
	  attachable_pragma = false;
	  break;
        
	case fcuda_PORTMERGE:
	  parse_fcuda_portmerge();
	  attachable_pragma = true;
	  break;

        case fcuda_COREINFO:
	  parse_fcuda_coreinfo();
	  attachable_pragma = true;
	  break;


	default : FcudaAnnotationParserError("Not Supported Construct");
      }
    }
    return attachable_pragma;
  }

  private static void parse_fcuda_grid()
  {
    Tools.println("FcudaAnnotationParser is parsing [GRID] annotation", 2);
    addToMap("fcuda", "grid");

    while (end_of_token() == false) 
    {
      String tokenVal[] = get_token().split("=");
      String clause = "token_" + tokenVal[0];
      Tools.println("clause=" + clause, 2);
      switch (fcuda_clause.valueOf(clause)) {
	case token_concur	:	parse_fcuda_concur(tokenVal[1]); break;
	case token_x_dim	:	parse_fcuda_xdim(tokenVal[1]); break;
	case token_y_dim	:	parse_fcuda_ydim(tokenVal[1]); break;
	case token_z_dim	:	parse_fcuda_zdim(tokenVal[1]); break;
	case token_name		:	parse_fcuda_name(tokenVal[1]); break;
	default : FcudaAnnotationParserError("NoSuchGridConstruct : " + clause);
      }
    }
  }


  private static void parse_fcuda_blocks()
  {
    Tools.println("FcudaAnnotationParser is parsing [BLOCKS] annotation", 2);
    addToMap("fcuda", "blocks");

    while (end_of_token() == false) 
    {
      String tokenVal[] = get_token().split("=");
      String clause = "token_" + tokenVal[0];
      Tools.println("clause=" + clause, 2);
      switch (fcuda_clause.valueOf(clause)) {
	case token_start_x	:	parse_fcuda_startx(tokenVal[1]); break;
	case token_end_x	:	parse_fcuda_endx(tokenVal[1]); break;
	case token_start_y	:	parse_fcuda_starty(tokenVal[1]); break;
	case token_end_y	:	parse_fcuda_endy(tokenVal[1]); break;
	default : FcudaAnnotationParserError("NoSuchBlocksConstruct : " + clause);
      }
    }
  }

  private static void parse_fcuda_sync()
  {
    Tools.println("FcudaAnnotationParser is parsing [SYNC] annotation", 2);
    addToMap("fcuda", "sync");

    while (end_of_token() == false) 
    {
      String tokenVal[] = get_token().split("=");
      String clause = "token_" + tokenVal[0];
      Tools.println("clause=" + clause, 2);
      switch (fcuda_clause.valueOf(clause)) {
	case token_type 	:	parse_fcuda_type(tokenVal[1]); break;
	case token_comp 	:	parse_fcuda_comp(); break;
	case token_trans	:	parse_fcuda_trans(); break;
	default : FcudaAnnotationParserError("NoSuchSyncConstruct : " + clause);
      }
    }
  }

  //##KG - #pragma FCUDA transfer type=burst cores=1/2/.. size=<bytes transferred> dir=0/1 pointer=<name> name=<name> begin/end
  //size = size per burst per core
  //dir (0/1 value):  0 => off-chip to on-chip 1=>reverse
  //pointer = list of array/pointer representing off-chip access [a1|a2 .. |an]
  //cores = number of cores for which to bring data (sequential)
  private static void parse_fcuda_transfer()
  {
    Tools.println("FcudaAnnotationParser is parsing [TRANSFER] annotation", 2);
    addToMap("fcuda", "transfer");

    while (end_of_token() == false) 
    {
      String tokenVal[] = get_token().split("=");
      String clause = "token_" + tokenVal[0];
      Tools.println("clause=" + clause, 2);
      switch (fcuda_clause.valueOf(clause)) {
	case token_type 	:	parse_fcuda_type(tokenVal[1]); break;
	case token_cores 	:	parse_fcuda_cores(tokenVal[1]); break;
	case token_size 	:	parse_fcuda_size(tokenVal[1]); break;
	case token_dir		:	parse_fcuda_dir(tokenVal[1]); break;
	case token_pointer	:	parse_fcuda_pointer(tokenVal[1]); break;
	case token_name 	:	parse_fcuda_name(tokenVal[1]); break;
	case token_unroll 	:	parse_fcuda_unroll(tokenVal[1]); break;
	case token_mpart 	:	parse_fcuda_mpart(tokenVal[1]); break;
	case token_array_split 	:	parse_fcuda_array_split(tokenVal[1]); break;
	case token_shape	:	parse_fcuda_shape(tokenVal[1]); break;
	case token_begin	:	parse_fcuda_begin(); break;
	case token_end		:	parse_fcuda_end(); break;
	case token_local	:	parse_fcuda_local(tokenVal[1]); break;
	case token_rsize	:	parse_fcuda_rsize(tokenVal[1]); break;
        case token_inline	:	parse_fcuda_inline(tokenVal[1]); break;
        case token_disable	:	parse_fcuda_disable(tokenVal[1]); break;
	default : FcudaAnnotationParserError("NoSuchTransferConstruct : " + clause);
      }
    }
  }

  private static void parse_fcuda_compute()
  {
    Tools.println("FcudaAnnotationParser is parsing [COMPUTE] annotation", 2);
    addToMap("fcuda", "compute");

    while (end_of_token() == false) 
    {
      String tokenVal[] = get_token().split("=");
      String clause = "token_" + tokenVal[0];
      Tools.println("clause=" + clause, 2);
      switch (fcuda_clause.valueOf(clause)) {
	case token_cores 	:	parse_fcuda_cores(tokenVal[1]); break;
	case token_name 	:	parse_fcuda_name(tokenVal[1]); break;
	case token_unroll 	:	parse_fcuda_unroll(tokenVal[1]); break;
	case token_mpart 	:	parse_fcuda_mpart(tokenVal[1]); break;
	case token_array_split 	:	parse_fcuda_array_split(tokenVal[1]); break;
	case token_shape	:	parse_fcuda_shape(tokenVal[1]); break;
	case token_begin	:	parse_fcuda_begin(); break;
	case token_end		:	parse_fcuda_end(); break;
        //Eric
        case token_shared       :       parse_fcuda_shared(tokenVal[1]); break;     
        case token_non_shared   :       parse_fcuda_non_shared(tokenVal[1]); break;
        case token_bram_core    :       parse_fcuda_bram_core(tokenVal[1]); break;
        case token_inline	:	parse_fcuda_inline(tokenVal[1]); break;
        case token_disable	:	parse_fcuda_disable(tokenVal[1]); break;
	default : FcudaAnnotationParserError("NoSuchComputeConstruct : " + clause);
      }
    }
  }

  private static void parse_fcuda_portmerge()
  {
    Tools.println("FcudaAnnotationParser is parsing [PORTMERGE] annotation", 2);
    addToMap("fcuda", "portmerge");

    while (end_of_token() == false) 
    {
      String tokenVal[] = get_token().split("=");
      String clause = "token_" + tokenVal[0];
      PrintTools.println("clause=" + clause, 2);
      switch (fcuda_clause.valueOf(clause)) {
	case token_remove_port_name 	:	parse_fcuda_remove_port_name(tokenVal[1]); break;
	case token_offset           	:	parse_fcuda_offset(tokenVal[1]); break;
        case token_port_core 	        :	parse_fcuda_port_core(tokenVal[1]); break;
	case token_port_id           	:	parse_fcuda_port_id(tokenVal[1]); break;
        case token_data_pack           	:	parse_fcuda_data_pack(tokenVal[1]); break;
	default : FcudaAnnotationParserError("NoSuchPortmergeConstruct : " + clause);
      }
    }
  }

  private static void parse_fcuda_coreinfo()
  {
    Tools.println("FcudaAnnotationParser is parsing [COREINFO] annotation", 2);
    addToMap("fcuda", "coreinfo");

    while (end_of_token() == false) 
    {
      String tokenVal[] = get_token().split("=");
      String clause = "token_" + tokenVal[0];
      PrintTools.println("clause=" + clause, 2);
      switch (fcuda_clause.valueOf(clause)) {
	case token_num_cores 	        :	parse_fcuda_num_cores(tokenVal[1]); break;
        case token_pipeline 	        :	parse_fcuda_pipeline(tokenVal[1]); break;
	default : FcudaAnnotationParserError("NoSuchCoreinfoConstruct : " + clause);
      }
    }
  }




  /** ---------------------------------------------------------------
   *
   *		A collection of parser routines for OpenMP clauses
   *
   * --------------------------------------------------------------- */

  private static void parse_fcuda_begin()
  {
    Tools.println("FcudaAnnotationParser is parsing [begin] clause", 2);

    addToMap("begin", "true");
    addToMap("end", "false");
  }

  private static void parse_fcuda_end()
  {
    Tools.println("FcudaAnnotationParser is parsing [end] clause", 2);
    addToMap("end", "true");
    addToMap("begin", "false");
  }

  private static void parse_fcuda_comp()
  {
    Tools.println("FcudaAnnotationParser is parsing [comp] clause", 2);
    match("(");
    Set<String> set = new HashSet<String>();
    parse_commaSeparatedList(set);
    match(")");

    addToMap("comp", set);
  }

  private static void parse_fcuda_trans()
  {
    Tools.println("FcudaAnnotationParser is parsing [trans] clause", 2);
    match("(");
    Set<String> set = new HashSet<String>();
    parse_commaSeparatedList(set);
    match(")");

    addToMap("trans", set);
  }

  private static void parse_fcuda_type(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [type] clause", 2);

    addToMap("type", val);
  }

  private static void parse_fcuda_cores(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [cores] clause", 2);

    addToMap("cores", val);
  }

  private static void parse_fcuda_unroll(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [unroll] clause", 2);

    addToMap("unroll", val);
  }

  private static void parse_fcuda_mpart(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [mpart] clause", 2);

    addToMap("mpart", val);
  }

  private static void parse_fcuda_array_split(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [array_split] clause", 2);

    addToMap("array_split", val);
  }

  private static void parse_fcuda_shape(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [shape] clause", 2);

    addToMap("shape", val);
  }

  private static void parse_fcuda_size(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [size] clause", 2);
    addToMap("size", val);
  }

  private static void parse_fcuda_dir(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [dir] clause", 2);
    addToMap("dir", val);
  }

  private static void parse_fcuda_pointer(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [pointer] clause", 2);
    addToMap("pointer", val);
  }

  private static void parse_fcuda_startx(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [start_x] clause", 2);

    addToMap("start_x", val);
  }

  private static void parse_fcuda_endx(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [end_x] clause", 2);

    addToMap("end_x", val);
  }

  private static void parse_fcuda_starty(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [start_y] clause", 2);

    addToMap("start_y", val);
  }

  private static void parse_fcuda_endy(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [end_y] clause", 2);

    addToMap("end_y", val);
  }

  private static void parse_fcuda_concur(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [concur] clause", 2);

    addToMap("concur", val);
  }

  private static void parse_fcuda_xdim(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [x_dim] clause", 2);

    addToMap("x_dim", val);
  }

  private static void parse_fcuda_ydim(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [y_dim] clause", 2);

    addToMap("y_dim", val);
  }

  private static void parse_fcuda_zdim(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [z_dim] clause", 2);

    addToMap("z_dim", val);
  }

  private static void parse_fcuda_name(String name)
  {
    Tools.println("FcudaAnnotationParser is parsing [name] clause", 2);

    addToMap("name", name);

  }

  private static void parse_fcuda_local(String varname)
  {
    Tools.println("FcudaAnnotationParser is parsing [local] clause", 2);

    addToMap("local", varname);

  }

  private static void parse_fcuda_rsize(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [rsize] clause", 2);

    addToMap("rsize", val);

  }

  private static void parse_fcuda_inline(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [inline] clause", 2);

    addToMap("inline", val);

  }

  private static void parse_fcuda_disable(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [inline] clause", 2);

    addToMap("disable", val);

  }
 
  private static void parse_fcuda_remove_port_name(String val)
  {
    PrintTools.println("FcudaAnnotationParser is parsing [remove_port_name] clause", 2);

    addToMap("remove_port_name", val);
  }

  private static void parse_fcuda_offset(String val)
  {
    PrintTools.println("FcudaAnnotationParser is parsing [offset] clause", 2);

    addToMap("offset", val);
  }

  private static void parse_fcuda_port_core(String val)
  {
    PrintTools.println("FcudaAnnotationParser is parsing [port_core] clause", 2);

    addToMap("port_core", val);
  }

  private static void parse_fcuda_port_id(String val)
  {
    PrintTools.println("FcudaAnnotationParser is parsing [port_id] clause", 2);

    addToMap("port_id", val);
  }

  private static void parse_fcuda_data_pack(String val)
  {
    PrintTools.println("FcudaAnnotationParser is parsing [data_pack] clause", 2);

    addToMap("data_pack", val);
  }


  private static void parse_fcuda_num_cores(String val)
  {
    PrintTools.println("FcudaAnnotationParser is parsing [num_cores] clause", 2);

    addToMap("num_cores", val);
  }

  private static void parse_fcuda_pipeline(String val)
  {
    PrintTools.println("FcudaAnnotationParser is parsing [pipeline] clause", 2);

    addToMap("pipeline", val);
  }

  //Eric
  private static void parse_fcuda_non_shared(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [nonshared] clause", 2);
   
    addToMap("non_shared", val);  
  }

  private static void parse_fcuda_bram_core(String val)
  {
    Tools.println("FcudaAnnotationParser is parsing [bram core] clause", 2);
     
    addToMap("bram_core", val); 
  }

  private static void parse_fcuda_shared(String val)
  {
     Tools.println("FcudaAnnotationParser is parsing [shared] clause", 2);
     
     addToMap("shared", val); 
  }

  /* This function reads a list of comma-separated variables
   * It checks the right parenthesis to end the parsing, but does not consume it
   */
  private static void parse_commaSeparatedList(Set<String> set)
  {
    String str = new String();
    for (;;) {
      // We have a nested paren string, parse it as one whole
      // If prev token wasn't a comma, we need to attach the previous str to the string
      // we're about to parse as one whole
      if ( check ("("))
      {
	str = str.concat(parse_ParenEnclosedExpr());
      }
      if ( check(")") )
      {
	if (str.length() != 0)
	  set.add(str);
	break;
      }
      if ( check(",") )
      {
	set.add(str);
	match(",");
	str = new String();
      }
      else if ( check("(") == true)
      {
	continue;
      }
      else
	str = str.concat(get_token());
    }
  }

  /*
   * This function parses a list of strings between a parenthesis, for example, 
   * expressions enclosed inside parenthesis
   */
  private static String parse_ParenEnclosedExpr()
  {
    String enclosed_str = new String();
    int paren_depth = 1;
    match("(");
    enclosed_str = "(";
    while (true) {
      if (check("(")) 
      {
	match("(");
	enclosed_str = enclosed_str.concat("(");
	paren_depth++;
      }
      else if (check(")")) 
      {
	if (paren_depth--==1)
	  break;
	match(")");
	enclosed_str = enclosed_str.concat(")");
      }
      else
      {
	enclosed_str = enclosed_str.concat(get_token());
      }
    }
    match(")");
    enclosed_str = enclosed_str.concat(")");
    return enclosed_str;
  }

  private static void FcudaAnnotationParserError(String text)
  {
    System.out.println("Fcuda Annotation Parser Syntax Error: " + text);
    System.out.println(display_tokens());
  }

  private static void addToMap(String key, String new_str)
  {
    if (fcuda_map.keySet().contains(key))
      Tools.exit("[Warning] Fcuda Annotation Parser detected duplicate pragma: " + key);
    else
      fcuda_map.put(key, new_str);
  }

  /* When a key already exists in the map */
  @SuppressWarnings("unchecked")
    private static void addToMap(String key, Set<String> new_set)
    {
      if (fcuda_map.keySet().contains(key))
      {
	Set<String> set = (Set<String>)fcuda_map.get(key);
	set.addAll(new_set);
      }
      else
      {
	fcuda_map.put(key, new_set);
      }
    }

  // reduction clause can be repeated as needed
  @SuppressWarnings("unchecked")
    private static void addToMap(String key, Map<String, Set<String>> new_map)
    {
      if (fcuda_map.keySet().contains(key))
      {
	Map<String, Set<String>> orig_map = (Map<String, Set<String>>)fcuda_map.get(key);
	for ( String new_str : new_map.keySet() )
	{
	  Set<String> new_set = new_map.get(new_str);
	  if (orig_map.keySet().contains(new_str))
	  {
	    Set<String> orig_set = orig_map.get(new_str);
	    orig_set.addAll(new_set);
	  }
	  else
	  {
	    orig_map.put(new_str, new_set);
	  }
	}
      }
      else
      {
	fcuda_map.put(key, new_map);
      }
    }

  public static enum fcuda_pragma
  {
      fcuda_GRID, 
      fcuda_BLOCKS, 
      fcuda_SYNC,
      fcuda_TRANSFER,
      fcuda_COMPUTE,
      fcuda_PORTMERGE,
      fcuda_COREINFO
  }

  public static enum fcuda_clause
  {
      token_concur,
      token_x_dim,
      token_y_dim,
      token_z_dim,
      token_begin,
      token_end,
      token_name,
      token_start_x,
      token_end_x,
      token_start_y,
      token_end_y,
      token_type,
      token_cores,
      token_unroll,
      token_mpart,
      token_array_split,
      token_shape,
      token_size,
      token_pointer,
      token_dir,
      token_comp,
      token_trans,
      token_local,
      token_lsize,
      token_rsize,
      token_remove_port_name,
      token_offset,
      token_port_core,
      token_port_id,
      token_data_pack,
      token_num_cores,
      token_pipeline,
      token_shared,
      token_non_shared,
      token_bram_core,
      token_inline,
      token_disable
  }
}
