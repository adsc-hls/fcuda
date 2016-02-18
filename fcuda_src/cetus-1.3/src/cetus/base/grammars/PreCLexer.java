// $ANTLR 2.7.7 (2006-11-01): "Pre.g" -> "PreCLexer.java"$

package cetus.base.grammars;


import java.io.InputStream;
import antlr.TokenStreamException;
import antlr.TokenStreamIOException;
import antlr.TokenStreamRecognitionException;
import antlr.CharStreamException;
import antlr.CharStreamIOException;
import antlr.ANTLRException;
import java.io.Reader;
import java.util.Hashtable;
import antlr.CharScanner;
import antlr.InputBuffer;
import antlr.ByteBuffer;
import antlr.CharBuffer;
import antlr.Token;
import antlr.CommonToken;
import antlr.RecognitionException;
import antlr.NoViableAltForCharException;
import antlr.MismatchedCharException;
import antlr.TokenStream;
import antlr.ANTLRHashString;
import antlr.LexerSharedInputState;
import antlr.collections.impl.BitSet;
import antlr.SemanticException;

@SuppressWarnings({"unchecked", "cast"})

public class PreCLexer extends antlr.CharScanner implements PreCParserTokenTypes, TokenStream
 {

	int openCount = 0;

public PreCLexer(InputStream in) {
	this(new ByteBuffer(in));
}
public PreCLexer(Reader in) {
	this(new CharBuffer(in));
}
public PreCLexer(InputBuffer ib) {
	this(new LexerSharedInputState(ib));
}
public PreCLexer(LexerSharedInputState state) {
	super(state);
	caseSensitiveLiterals = true;
	setCaseSensitive(true);
	literals = new Hashtable();
}

public Token nextToken() throws TokenStreamException {
	Token theRetToken=null;
tryAgain:
	for (;;) {
		Token _token = null;
		int _ttype = Token.INVALID_TYPE;
		resetText();
		try {   // for char stream error handling
			try {   // for lexical error handling
				if ((LA(1)=='#') && (LA(2)=='i') && (LA(3)=='n')) {
					mInclude(true);
					theRetToken=_returnToken;
				}
				else if ((LA(1)=='#') && ((LA(2) >= '\u0003' && LA(2) <= '\u00ff')) && (true)) {
					mPreprocDirective(true);
					theRetToken=_returnToken;
				}
				else if (((LA(1) >= '\u0003' && LA(1) <= '\u00ff')) && (true) && (true)) {
					mRest(true);
					theRetToken=_returnToken;
				}
				else if ((LA(1)=='{') && (true) && (true)) {
					mLcurly(true);
					theRetToken=_returnToken;
				}
				else if ((LA(1)=='}') && (true) && (true)) {
					mRcurly(true);
					theRetToken=_returnToken;
				}
				else if ((LA(1)=='\n'||LA(1)=='\r') && (true) && (true)) {
					mNewline(true);
					theRetToken=_returnToken;
				}
				else {
					if (LA(1)==EOF_CHAR) {uponEOF(); _returnToken = makeToken(Token.EOF_TYPE);}
				else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
				}
				
				if ( _returnToken==null ) continue tryAgain; // found SKIP token
				_ttype = _returnToken.getType();
				_ttype = testLiteralsTable(_ttype);
				_returnToken.setType(_ttype);
				return _returnToken;
			}
			catch (RecognitionException e) {
				throw new TokenStreamRecognitionException(e);
			}
		}
		catch (CharStreamException cse) {
			if ( cse instanceof CharStreamIOException ) {
				throw new TokenStreamIOException(((CharStreamIOException)cse).io);
			}
			else {
				throw new TokenStreamException(cse.getMessage());
			}
		}
	}
}

	public final void mPreprocDirective(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = PreprocDirective;
		int _saveIndex;
		
		match('#');
		{
		boolean synPredMatched7 = false;
		if (((LA(1)=='#') && (LA(2)=='i') && (LA(3)=='n'))) {
			int _m7 = mark();
			synPredMatched7 = true;
			inputState.guessing++;
			try {
				{
				match("include");
				}
			}
			catch (RecognitionException pe) {
				synPredMatched7 = false;
			}
			rewind(_m7);
inputState.guessing--;
		}
		if ( synPredMatched7 ) {
			mInclude(false);
		}
		else if (((LA(1) >= '\u0003' && LA(1) <= '\u00ff')) && (true) && (true)) {
			mRest(false);
		}
		else {
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mInclude(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Include;
		int _saveIndex;
		
		match("#include");
		mRest(false);
		if ( inputState.guessing==0 ) {
			
			
				if(openCount != 0) {
					String text = getText();
					setText("internal"+text);
				}
			
			
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mRest(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Rest;
		int _saveIndex;
		
		{
		_loop12:
		do {
			switch ( LA(1)) {
			case '{':
			{
				mLcurly(false);
				break;
			}
			case '}':
			{
				mRcurly(false);
				break;
			}
			default:
				if ((_tokenSet_0.member(LA(1)))) {
					{
					match(_tokenSet_0);
					}
				}
			else {
				break _loop12;
			}
			}
		} while (true);
		}
		mNewline(false);
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mLcurly(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Lcurly;
		int _saveIndex;
		
		match('{');
		if ( inputState.guessing==0 ) {
			openCount ++;
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mRcurly(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Rcurly;
		int _saveIndex;
		
		match('}');
		if ( inputState.guessing==0 ) {
			openCount --;
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	public final void mNewline(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Newline;
		int _saveIndex;
		
		{
		if ((LA(1)=='\r') && (LA(2)=='\n')) {
			match("\r\n");
		}
		else if ((LA(1)=='\n')) {
			match('\n');
		}
		else if ((LA(1)=='\r') && (true)) {
			match('\r');
		}
		else {
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		
		}
		if ( inputState.guessing==0 ) {
			newline();
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mSpace(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Space;
		int _saveIndex;
		
		{
		switch ( LA(1)) {
		case ' ':
		{
			match(' ');
			break;
		}
		case '\t':
		{
			match('\t');
			break;
		}
		case '\u000c':
		{
			match('\014');
			break;
		}
		default:
		{
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		}
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	
	private static final long[] mk_tokenSet_0() {
		long[] data = new long[8];
		data[0]=-9224L;
		data[1]=-2882303761517117441L;
		for (int i = 2; i<=3; i++) { data[i]=-1L; }
		return data;
	}
	public static final BitSet _tokenSet_0 = new BitSet(mk_tokenSet_0());
	
	}
