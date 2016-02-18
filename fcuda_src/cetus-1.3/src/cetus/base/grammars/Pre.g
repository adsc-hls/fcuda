header {
package cetus.base.grammars;

}

{
import java.io.*;
@SuppressWarnings({"unchecked", "cast"})
}
class PreCParser extends Parser;

options
        {
        k = 2;
        //exportVocab = PreC;
        //buildAST = true;
        //ASTLabelType = "TNode";

        // Copied following options from java grammar.
        codeGenMakeSwitchThreshold = 2;
        codeGenBitsetTestThreshold = 3;
        }

{

}


programUnit [PrintStream out,String filename]
        {
			// set line 1 and original file name
			out.println("#line 1 \""+filename+"\"");
		}
		:
		(
			in:Include
			{
					String s = null;
					s = in.getText();

					// non global include clause
					if(s.startsWith("internal")){
						out.print(s.substring(8));
					}
					// global include clause
					else{
					    // marker for start of header inclusion
						out.print("#pragma startinclude "+in.getText());
						// adjust line numbering
						out.println("#line "+in.getLine());
						//The actual include clause
						out.print(in.getText());
						// marker for end of header inclusion
						out.println("#pragma endinclude");
						// adjust line numbering
						out.println("#line "+(in.getLine()+1));

					}

			}

		| 	pre:PreprocDirective
			{
				out.print(pre.getText());
			}

		|	re:Rest
			{
				out.print(re.getText());
			}
		)+
		;

{
@SuppressWarnings({"unchecked", "cast"})
}


class PreCLexer extends Lexer;

options
        {
        k = 3;
        //exportVocab = PreC;
        //testLiterals = false;
        charVocabulary = '\3'..'\377';
        }



{
	int openCount = 0;

}

PreprocDirective :

        '#'
        (
        	( "include" ) => Include
			| Rest
		)
		;

Include
        :
        "#include" Rest
        {

        	if(openCount != 0) {
        		String text = getText();
        		setText("internal"+text);
        	}

        }

        ;

Rest
		:
			(
				~( '\n' | '\r' | '{' | '}')
				| Lcurly
				| Rcurly
			)*
		 	Newline

		;


Newline
        :       (
				"\r\n"
                | '\n'
				| '\r'
                )
                {newline();}
        ;

protected  Space:
        ( ' ' | '\t' | '\014')
        ;

Lcurly
		: '{'	{ openCount ++;}
		;
Rcurly
		: '}'   { openCount --;}
		;
