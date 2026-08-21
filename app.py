import io,re
from pathlib import Path
from collections import Counter
import pandas as pd, streamlit as st, spacy, hanlp, pyphen
from docx import Document
from lexical_diversity import lex_div as ld

st.set_page_config(page_title='Bilingual Essay NLP Analyser',layout='wide')
EN_CONN=set('and but or so yet because although though while whereas if unless since when before after therefore however moreover furthermore nevertheless consequently thus hence instead otherwise similarly likewise additionally also'.split())
ZH_CONN=set('和 与 及 而 但 但是 可是 然而 因为 因此 所以 虽然 尽管 如果 假如 除非 当 同时 此外 而且 并且 另外 总之 例如 由於 因為 雖然 儘管 當'.split())
EN_MODAL=set('can could may might must shall should will would ought'.split())
ZH_MODAL=set('能 能够 能夠 可以 可 会 會 应 应该 應 應該 须 須 必须 必須 得 要 愿意 願意 敢'.split())
SFP=set('啊 呀 吧 呢 吗 嗎 嘛 啦 喇 咯 囉 哦 呦 了 啫 呗 唄 哩'.split())
P={'1s':set('i me my mine myself 我 本人'.split()),'1p':set('we us our ours ourselves 我们 我們 咱们 咱們'.split()),'2':set('you your yours yourself yourselves 你 您 你们 你們'.split()),'3s':set('he him his himself she her hers herself it its itself 他 她 它 其'.split()),'3p':set('they them their theirs themselves 他们 她们 它们 他們 她們 它們'.split())}
D=pyphen.Pyphen(lang='en_US')
def rate(a,b): return a/b if b else 0
def pct(a,b): return rate(a,b)*100
def syll(w):
 w=re.sub('[^A-Za-z]','',w); return len(D.inserted(w).split('-')) if w else 0
def read(f):
 b=f.getvalue()
 if f.name.lower().endswith('.docx'): return '\n'.join(p.text for p in Document(io.BytesIO(b)).paragraphs)
 for e in ['utf-8-sig','utf-8','gb18030','big5']:
  try:return b.decode(e)
  except UnicodeDecodeError:pass
 return b.decode('utf-8',errors='replace')
def chinese(t):
 a=[c for c in t if c.isalpha()]; return rate(sum('\u4e00'<=c<='\u9fff' for c in a),len(a))>=.2
def zsent(t): return [x.strip() for x in re.split(r'(?<=[。！？!?；;])\s*|\n+',t) if x.strip()]
@st.cache_resource
def en(): return spacy.load('en_core_web_sm')
@st.cache_resource
def zh(): return (hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH),hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL))
def wl(f):
 if not f:return set()
 try:
  x=pd.read_excel(io.BytesIO(f.getvalue())) if f.name.endswith(('xlsx','xls')) else pd.read_csv(io.BytesIO(f.getvalue()),header=None)
  return {str(v).strip().lower() for v in x.astype(str).stack() if str(v)!='nan'}
 except:return set()
def common(words,sents,paras,nouns,verbs,conns,mods,nps,pas,cla,rel,sub,agr,ngsl,nawl,iszh,sfps):
 w=[x.lower() for x in words]; n=len(w); ns=len(sents); c=[x for x in w if x in conns]; m=[x for x in w if x in mods]
 cnt=lambda s:sum(x in s for x in w); p1s,p1p,p2,p3s,p3p=[cnt(P[k]) for k in ['1s','1p','2','3s','3p']]
 sy=[] if iszh else [syll(x) for x in w]
 out={'MTLD':ld.mtld(w) if len(w)>=10 else None,'Mean Utt. Len.':rate(n,ns),'Sentences':ns,'TTR':rate(len(set(w)),n),'Tokens':n,'Types':len(set(w)),'Complex Word %':None if iszh else pct(sum(x>=3 for x in sy),n),'Syllables/Sent':None if iszh else rate(sum(sy),ns),'Syllables/Word':None if iszh else rate(sum(sy),n),'Connectives %':pct(len(c),n),'Connectives Freq':len(c),'Connectives Type':len(set(c)),'Connectives TTR':rate(len(set(c)),len(c)),'Modal %':pct(len(m),n),'Modal Freq':len(m),'Modal/Sent':rate(len(m),ns),'Mean Noun Syll.':None if iszh else rate(sum(syll(x) for x in nouns),len(nouns)),'Noun %':pct(len(nouns),n),'Noun Freq':len(nouns),'Noun TTR':rate(len(set(nouns)),len(nouns)),'Noun/Sent':rate(len(nouns),ns),'Unique Nouns':len(set(nouns)),'Mean Verb Syll.':None if iszh else rate(sum(syll(x) for x in verbs),len(verbs)),'Unique Verbs':len(set(verbs)),'Verb %':pct(len(verbs),n),'Verb Freq':len(verbs),'Verb TTR':rate(len(set(verbs)),len(verbs)),'Verb/Sent':rate(len(verbs),ns),'NAWL Cov%':pct(sum(x in nawl for x in w),n) if nawl and not iszh else None,'NGSL Cov%':pct(sum(x in ngsl for x in w),n) if ngsl and not iszh else None}
 for label,num in [('1st Person',p1s+p1p),('1st Plural',p1p),('1st Singular',p1s),('2nd Person',p2),('3rd Person',p3s+p3p),('3rd Plural',p3p),('3rd Singular',p3s)]:out[label+' %']=pct(num,n);out[label+' Freq']=num
 for x in ['i','me','us','we','you']:out[x.title()+' %']=pct(w.count(x),n);out[x.title()+' Freq']=w.count(x)
 out.update({'Noun Phrase Count':len(nps),'Noun Phrase Length':rate(sum(nps),len(nps)),'Noun Phrase/Sent':rate(len(nps),ns),'Subject-Verb Agree/Sent':rate(agr,ns),'Passive Count':pas,'Passive/Sent':rate(pas,ns),'Clause Count':cla,'Clauses/Sent':rate(cla,ns),'Relative Clause Count':rel,'Relative Clauses/Sent':rate(rel,ns),'Subordinate Clause Count':sub,'Subordinate Clauses/Sent':rate(sub,ns),'Paragrahs per Text':len(paras),'Sentences per Paragraph':rate(ns,len(paras)),'Tokens per Paragraph':rate(n,len(paras)),'Sentence Final Particle Count':sfps if iszh else None,'Sentence Final Particles/Sent':rate(sfps,ns) if iszh else None})
 return out
def proc_en(t,ng,nw):
 d=en();doc=d(t);ss=list(doc.sents);rows=[[i,s.text.strip(),x.text,x.pos_] for i,s in enumerate(ss,1) for x in s if not x.is_space];a=[x for x in doc if x.is_alpha];w=[x.text for x in a];no=[x.lemma_.lower() for x in a if x.pos_ in ('NOUN','PROPN')];ve=[x.lemma_.lower() for x in a if x.pos_ in ('VERB','AUX')];np=[sum(x.is_alpha for x in z) for z in doc.noun_chunks];pas=sum(x.dep_ in ('auxpass','nsubjpass') for x in doc);rel=sum(x.dep_=='relcl' for x in doc);sub=sum(x.dep_ in ('advcl','ccomp','xcomp','acl') for x in doc);cl=len({x.i for x in doc if x.pos_ in ('VERB','AUX') and x.dep_ in ('ROOT','conj','advcl','ccomp','xcomp','acl','relcl')});agr=sum(bool(set(x.morph.get('Number'))&set(x.head.morph.get('Number'))) for x in doc if x.dep_ in ('nsubj','nsubjpass'));p=[x for x in t.splitlines() if x.strip()];return pd.DataFrame(rows,columns=['Sentence ID','Sentence','Word','POS Tag']),common(w,ss,p,no,ve,EN_CONN,EN_MODAL,np,pas,cl,rel,sub,agr,ng,nw,False,0)
def proc_zh(t,ng,nw):
 tok,pos=zh();ss=zsent(t);ws=tok(ss);ts=pos(ws);rows=[[i,s,w,g] for i,(s,aa,bb) in enumerate(zip(ss,ws,ts),1) for w,g in zip(aa,bb)];pairs=[(w,g) for aa,bb in zip(ws,ts) for w,g in zip(aa,bb) if re.search('[\u3400-\u9fffA-Za-z0-9]',w)];w=[x[0] for x in pairs];tags=[x[1] for x in pairs];no=[x for x,g in pairs if g.startswith('N')];ve=[x for x,g in pairs if g.startswith('V')];nps=[]
 for bb in ts:
  run=0
  for g in bb+['STOP']:
   if g.startswith(('N','D','JJ','PN','M','CD','OD')):run+=1
   elif run:nps.append(run);run=0
 pas=sum(x in {'被','让','讓','给','給','遭','受'} for x in w);rel=sum(x=='的' and i and tags[i-1].startswith('V') for i,x in enumerate(w));sub=sum(x in ZH_CONN for x in w);cl=sum(g.startswith('V') for g in tags);sf=sum(bool(a) and a[-1] in SFP for a in ws);p=[x for x in t.splitlines() if x.strip()];return pd.DataFrame(rows,columns=['Sentence ID','Sentence','Word','POS Tag']),common(w,ss,p,no,ve,ZH_CONN,ZH_MODAL,nps,pas,cl,rel,sub,0,ng,nw,True,sf)
def sname(n,used):
 x=re.sub(r'[\\/*?:\[\]]','_',Path(n).stem)[:31] or 'Sheet';base=x;i=2
 while x in used:x=base[:28]+f'_{i}';i+=1
 used.add(x);return x
st.title('Bilingual Essay NLP Analyser')
st.write('Upload English or Simplified/Traditional Chinese essays and export word/POS and metric workbooks.')
with st.sidebar:
 ngf=st.file_uploader('Optional official NGSL list',type=['csv','xlsx','xls']);nwf=st.file_uploader('Optional official NAWL list',type=['csv','xlsx','xls']);st.info('No application-level file-count cap. Browser, RAM, upload-size and timeout limits still apply.')
fs=st.file_uploader('Essay files',type=['txt','docx'],accept_multiple_files=True)
if st.button('Analyse',type='primary',disabled=not fs):
 ng,nw=wl(ngf),wl(nwf);posout=[];metrics=[];errs=[]
 for f in fs:
  try:
   t=read(f);zhflag=chinese(t);df,m=proc_zh(t,ng,nw) if zhflag else proc_en(t,ng,nw);posout.append((f.name,df));metrics.append({'File Name':f.name,'Language':'Chinese' if zhflag else 'English',**m})
  except Exception as e:errs.append([f.name,str(e)])
 if posout:
  a=io.BytesIO();used=set()
  with pd.ExcelWriter(a,engine='xlsxwriter') as x:
   for n,d in posout:d.to_excel(x,sheet_name=sname(n,used),index=False)
  md=pd.DataFrame(metrics);num=md.select_dtypes('number');summary=pd.DataFrame([{**{'File Name':'','Language':''},**{c:None for c in num}},{**{'File Name':'Mean','Language':''},**num.mean().to_dict()},{**{'File Name':'Std. Dev.','Language':''},**num.std(ddof=1).to_dict()}]);final=pd.concat([md,summary],ignore_index=True);b=io.BytesIO()
  with pd.ExcelWriter(b,engine='xlsxwriter') as x:
   final.to_excel(x,sheet_name='Metrics',index=False);pd.DataFrame({'Note':['English: spaCy UPOS/dependencies.','Chinese: HanLP tokenization and CTB9 POS.','Chinese syntactic counts are rule-based estimates and require validation.','Chinese syllable and English-only vocabulary coverage cells are blank.','Standard deviation is sample SD (ddof=1).']}).to_excel(x,sheet_name='Methodology',index=False)
  st.download_button('Download word and POS workbook',a.getvalue(),'Essays_Words_POS.xlsx');st.download_button('Download linguistic metrics workbook',b.getvalue(),'Linguistic_Metrics.xlsx');st.dataframe(md)
 if errs:st.error(pd.DataFrame(errs,columns=['File','Error']))
