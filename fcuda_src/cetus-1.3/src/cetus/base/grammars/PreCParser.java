// $ANTLR 2.7.7 (2006-11-01): "Pre.g" -> "PreCParser.java"$

package cetus.base.grammars;


import antlr.TokenBuffer;
import antlr.TokenStreamException;
import antlr.TokenStreamIOException;
import antlr.ANTLRException;
import antlr.LLkParser;
import antlr.Token;
import antlr.TokenStream;
import antlr.RecognitionException;
import antlr.NoViableAltException;
import antlr.MismatchedTokenException;
import antlr.SemanticException;
import antlr.ParserSharedInputState;
import antlr.collections.impl.BitSet;

import java.io.*;
@SuppressWarnings({"unchecked", "cast"})

public class PreCParser extends antlr.LLkParser       implements PreCParserTokenTypes
 {



protected PreCParser(TokenBuffer tokenBuf, int k) {
  super(tokenBuf,k);
  tokenNames = _tokenNames;
}

public PreCParser(TokenBuffer tokenBuf) {
  this(tokenBuf,2);
}

protected PreCParser(TokenStream lexer, int k) {
  super(lexer,k);
  tokenNames = _tokenNames;
}

public PreCParser(TokenStream lexer) {
  this(lexer,2);
}

public PreCParser(ParserSharedInputState state) {
  super(state,2);
  tokenNames = _tokenNames;
}

	public final void programUnit(
		PrintStream out,String filename
	) throws RecognitionException, TokenStreamException {
		
		Token  in = null;
		Token  pre = null;
		Token  re = null;
		
					// set line 1 and original file name
					out.println("#line 1 \""+filename+"\"");
				
		
		try {      // for error handling
			{
			int _cnt3=0;
			_loop3:
			do {
				switch ( LA(1)) {
				case Include:
				{
					in = LT(1);
					match(Include);
					
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
					
								
					break;
				}
				case PreprocDirective:
				{
					pre = LT(1);
					match(PreprocDirective);
					
									out.print(pre.getText());
								
					break;
				}
				case Rest:
				{
					re = LT(1);
					match(Rest);
					
									out.print(re.getText());
								
					break;
				}
				default:
				{
					if ( _cnt3>=1 ) { break _loop3; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				}
				_cnt3++;
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			reportError(ex);
			recover(ex,_tokenSet_0);
		}
	}
	
	
	public static final String[] _tokenNames = {
		"<0>",
		"EOF",
		"<2>",
		"NULL_TREE_LOOKAHEAD",
		"Include",
		"PreprocDirective",
		"Rest",
		"Newline",
		"Space",
		"Lcurly",
		"Rcurly"
	};
	
	private static final long[] mk_tokenSet_0() {
		long[] data = { 2L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_0 = new BitSet(mk_tokenSet_0());
	
	}
