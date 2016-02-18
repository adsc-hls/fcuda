package cetus.exec;

import cetus.base.grammars.NewCLexer;
import cetus.base.grammars.NewCParser;
import cetus.base.grammars.PreCLexer;
import cetus.base.grammars.PreCParser;
import cetus.hir.Declaration;
import cetus.hir.TranslationUnit;
import cetus.hir.Tools;

import java.io.*;
import java.util.Vector; // not supported
import java.util.Arrays;

public class Parser{

  /** This nested class continuously reads from the output
   * or error stream of an external parser and reproduces
   * the data on the JVM's output or error stream.
   */
  private class PipeThread extends Thread
  {
    private BufferedReader source;
    private PrintStream dest;

    PipeThread(BufferedReader source, PrintStream dest)
    {
      this.source = source;
      this.dest = dest;
    }

    public void run()
    {
      String s;
      try {
        while ((s = source.readLine()) != null)
          dest.println(s);
      } catch (IOException e) {
        dest.println("cetus: I/O error on redirection, " + e);
      }
    }
  }

  /* Working directory for the parser. */
  private File pwd;

  /**
  * Constructs a new parser that works in the current directory.
  */
  public Parser()
  {
    pwd = new File(System.getProperty("user.dir"));
  }

  /**
  * Constructs a new parser that works in the specified directory.
  * @param dir the working directory.
  */
  public Parser(String dir)
  {
    pwd = new File(dir);
  }

  /**
   * Parse this translation unit.
   *
   * @param input_filename Name of file to parse
   * @throws IOException if there is a problem accessing any file.
   * @return Parsed TranslationUnit
   */
  public TranslationUnit parse(String input_filename) throws IOException
  {
		return parseAntlr(input_filename);
/**
	* If you want to add another parser other than antlr, then you need to modify
	* parseExternal routine below for your purpose and update the cetus manual
	*
    return parseExternal(input_filename);
*/
  }

  /**
   * Parse the associated input file using the Antlr
   * parser and create IR for this translation unit.
   * Parsing is performed in three stages
   *   Stage 1:
   *     File is annotated with pragmas to mark start and end of
   *     file inclusion (PreCParser and PreCLexer)
   *   Stage 2:
   *     File is run through an external C preprocessor
   *   Stage 3:
   *     Proprocessed files is feed into actual Parser
   *     (NewCParser and NewCLexer)
   * All classes are dynamically loadedto avoid unwanted
   * false dependency when other parsers are used.
   *
   * @param input_filename Name of file to parse
   * @return Parsed TranslationUnit
   */
  @SuppressWarnings({"unchecked", "cast"})
  protected TranslationUnit parseAntlr(String input_filename)
  {
		TranslationUnit tu = new TranslationUnit(input_filename);
    String filename = null;
    File prefile;
    byte[] barray = null;
      //InputStream source = null;
      // pre step to handle header files
      // Insert markers for start and end of a header file
    String prename = null;

    /* Create the Antlr-derived lexer and parser through the ClassLoader
       so antlr.jar will be required only if the Antlr parser is used. */
    Class[] params = new Class[1];
    Object[] args = new Object[1];

    ClassLoader loader = getClass().getClassLoader();
    //ClassLoader loader = getClass().getClassLoader();
    Class class_TokenStream = null;
    try  {
      Class class_PreCLexer  = loader.loadClass("cetus.base.grammars.PreCLexer");
      class_TokenStream = loader.loadClass("antlr.TokenStream");
      params[0] = InputStream.class;
      args[0] = new DataInputStream(new FileInputStream(input_filename));
      PreCLexer lexer = (PreCLexer)class_PreCLexer.getConstructor(params).newInstance(args);

      Class class_PreCParser = loader.loadClass("cetus.base.grammars.PreCParser");
      params[0] = class_TokenStream;
      args[0] = lexer;
      PreCParser parser = (PreCParser)class_PreCParser.getConstructor(params).newInstance(args);

      filename = (new File(input_filename)).getName();
      prename = "cppinput_" + filename;
      prefile = new File(pwd, prename);
      prefile.deleteOnExit();
      FileOutputStream fo = new FileOutputStream(prefile);
      // Add option to print pre annotated input file before
      // calling external preprocessor and exit
      if(Driver.getOptionValue("debug_preprocessor_input")!=null)
      {
        parser.programUnit(System.out,filename);
        fo.close();
        Tools.exit(0);
      }

      parser.programUnit(new PrintStream(fo),filename);

      fo.close();
    } catch (ClassNotFoundException e) {
      System.err.println("cetus: could not load class " + e);
      Tools.exit(1);
    } catch (NoSuchMethodException e) {
      System.err.println("cetus: could not find constructor " + e);
      Tools.exit(1);
    } catch (IllegalAccessException e) {
      System.err.println("cetus: could not access constructor " + e);
      Tools.exit(1);
    } catch (InstantiationException e) {
      System.err.println("cetus: constructor failed " + e);
      Tools.exit(1);
    } catch (FileNotFoundException e) {
      System.err.println("cetus: could not read input file " + e);
      Tools.exit(1);
    } catch (IOException e) {
      System.err.println("cetus: could not create intermdiate output file " + e);
      Tools.exit(1);
    } catch (Exception e) {
      System.err.println("cetus: exception: " + e);
      e.printStackTrace();
      Tools.exit(1);
    }

    // Run cpp on the input file and output to a temporary file.
    try {
      ByteArrayOutputStream bo = new ByteArrayOutputStream(50000);
      PrintStream outStream = new PrintStream(bo);

			String cmd = Driver.getOptionValue("preprocessor");
			cmd += getMacros() + " " + prename;
      ProcessBuilder pb =
          new ProcessBuilder(Arrays.asList(cmd.trim().split(" +")));
      pb.directory(pwd);
      Process p = pb.start();

      BufferedReader inReader = new BufferedReader(new InputStreamReader(p.getInputStream()));
      BufferedReader errReader = new BufferedReader(new InputStreamReader(p.getErrorStream()));

      PipeThread out_pipe = new PipeThread(inReader, outStream);
      PipeThread err_pipe = new PipeThread(errReader, System.err);

      out_pipe.start();
      err_pipe.start();

      if (p.waitFor() != 0)
      {
        System.err.println("cetus: preprocessor terminated with exit code " + p.exitValue());
        Tools.exit(1);
      }

      out_pipe.join();
      err_pipe.join();

      barray = bo.toByteArray();

      //----------------
      //System.out.write(barray, 0, Array.getLength(barray));

    } catch (java.io.IOException e) {
      System.err.println("Fatal error creating temporary file: " + e);Tools.exit(1);
    } catch (java.lang.InterruptedException e) {
      System.err.println("Fatal error starting preprocessor: " + e);Tools.exit(1);
    }

    // Add option to print preprocessed input file before
    // calling parser and exit
    if(Driver.getOptionValue("debug_parser_input")!=null)
    {
        System.out.println("Dumping parser input file");
        String str = new String(barray);
        System.out.println(str);
        Tools.exit(0);
    }
    // Actual antlr parser is called

    try {
      Class class_NewCLexer = loader.loadClass("cetus.base.grammars.NewCLexer");
      params[0] = InputStream.class;
      args[0] = new DataInputStream(new ByteArrayInputStream(barray));
      NewCLexer lexer = (NewCLexer)class_NewCLexer.getConstructor(params).newInstance(args);

      lexer.setOriginalSource(filename);
      lexer.setTokenObjectClass("cetus.base.grammars.CToken");
      lexer.initialize();

      Class class_NewCParser = loader.loadClass("cetus.base.grammars.NewCParser");
      params[0] = class_TokenStream;
      args[0] = lexer;
      NewCParser parser = (NewCParser)class_NewCParser.getConstructor(params).newInstance(args);

      parser.getPreprocessorInfoChannel(lexer.getPreprocessorInfoChannel());
      parser.setLexer(lexer);
      parser.translationUnit(tu);
    } catch (Exception e) {
      System.err.println("Parse error: " + e);
      e.printStackTrace();
      Tools.exit(1);
    }

/*
    try {
      Class[] pparams = new Class[2];
      pparams[0] = TranslationUnit.class;
      pparams[1] = OutputStream.class;
      pparams[1] = PrintWriter.class;
      tu.setPrintMethod(pparams[0].getMethod("defaultPrint2", pparams));
    } catch (NoSuchMethodException e) {
      throw new InternalError();
    }
*/
		return tu;
  }

	// Reads option value from -macro and returns a converted string to be added
	// in the preprocessor cmd line.
	protected static String getMacros()
	{
		String ret = " ";
		String macro = Driver.getOptionValue("macro");
		if ( macro == null )
			return ret;

		String[] macro_list = macro.split(",");
		for ( int i=0; i<macro_list.length; i++ )
			ret += (" -D"+macro_list[i]);

		return ret;
	}
}
