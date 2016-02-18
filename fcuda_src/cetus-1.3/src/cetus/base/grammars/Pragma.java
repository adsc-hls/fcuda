package cetus.base.grammars;

	public class Pragma {
		String str = null;
		int type = 0;
		public static final int comment = 0;
		public static final int pragma = 1;
		public Pragma (String s,int t){
			type = t;
			str = s;
		} 	
	}