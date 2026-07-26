Integrace ChatGPT a V271  
Tato část je záměrně evidence-first a protocol -first. Neopírám ji o představu neveřejných 
API nebo neohlášených partnerství s OpenAI. Vycházím z toho, co je veřejně vidět: ChatGPT agent, connectors, Apps in ChatGPT, Apps SDK, Responses API a remote MCP support. Na straně V271 vycházím z vašich interních materiálů a z této konverzace; to 
znamená, že pro potřeby plánování předpokládám B2C chat, dokumentovou 
knihovnu/RAG, agentní vrstvu, voice/avatar line a více distribučních surfaces. Tyto detaily nejsou v této zprávě externě ověřeny a je potřeba je interně potvrdit.  [23] 
Nejrozumnější integrace není „ChatGPT nahradit“ ani „ChatGPT se podřídit“. 
Nejrozumnější je vztah platforma + continuity shell . ChatGPT může být v řadě scénářů 
hlavní frontier runtime pro reasoning a general -purpose agentní práci; V271 pak může být 
dlouhodobý shell pro osobní data, dokumenty, preferenční kontinuitu, evropskou správu 
souhlasu, trust controls a později i monetizaci tvůrců agentů. To je model podobný tomu, 
jak některé firmy stavějí na LLM poskytovateli, ale vlastní identitu, workflow, data graph a 
ekonomiku. OpenAI veřejně podporuje takovou logiku přes Apps SDK a agentní nástroje pro třetí strany. [24]  
Prakticky doporučuji pět integračních bodů.  
 
 
Tento diagram je návrhový, nikoli veřejně potvrzená architektura OpenAI. Veřejně je však doložitelné, že OpenAI podporuje agentní tool use, connectors, Apps in ChatGPT a remote MCP; proto je protocol -first interop realistický směr. [25]  
První integrační bod je V271 MCP Gateway. V271 by mělo umět zpřístupnit svou 
dokumentovou knihovnu, výzkumné výsledky, agentní capability cards a případně osobní task objects přes standardní bránu, kterou lze připojit do cizího agentního runtime. Tím se V271 nestává závislým na jediném UI a současně může fungovat jako persistentní knowledge layer. Tento směr je vysoce pravděpodobný, protože MCP už je veřejně prosazovaný jako otevřený protokol pro kontext a nástroje. [26] 
Druhý integrační bod je research object exchange . Když uživatel spustí deep research 
nebo agentní úlohu v ChatGPT, výsledkem by neměl být jen chat transcript. V271 by mělo 
umět tyto výstupy převést na strukturované objekty: briefing, source pack, workflow draft, 
tabulku rozhodnutí, souhrn s citacemi a navazující agentní úlohy. To odpovídá trendu, kdy ChatGPT produkuje výzkumné a akční artefakty, a posouvá V271 do role osobního „systému záznamu“ pro AI práci. [27]  
Třetí integrační bod je identity a continuity bridge . Tady je nutné být opatrný: veřejně není 
oznámeno, že by OpenAI budovalo plnohodnotnou federaci identit s externími consumer 

aplikacemi. Proto doporučuji neplánovat magický „single memory layer“ napříč ChatGPT a 
V271. Místo toho dává smysl stavět na exportovatelných profiles, explicitních consent scopes, receipts a později na verifiable credentials. Jinými slovy, ne synchronizovat vše, ale synchronizovat pouze to, co uživatel vědomě povolí, v auditovatelné podobě. [28] 
Čtvrtý integrační bod je federovaný Agent Store . Veřejný stav OpenAI dovoluje mluvit o 
GPT Store a Apps in ChatGPT; nedovoluje tvrdit, že OpenAI otevře společný katalog třetím stranám v podobě, která by dovolila plně sdílený store se V271. Přesto je strategicky rozumné připravit V271 tak, aby mělo vlastní agent card schema, reputační metadata, 
permission manifests a pricing model, které budou použitelné jak ve V271, tak v budoucí 
federaci s dalšími marketplace surfaces. To je „vize s medium confidence“. [29]  
Pátý integrační bod je distribution arbitrage . ChatGPT může být pro V271 distribuční kanál 
pro akvizici a discovery, zatímco V271 native web, desktop a mobile mohou být místa, kde 
se odehrává hlubší dlouhodobá kontinuita, citlivější workflow a monetizace. Tento model 
je podobný tomu, jak aplikace používají sociální sítě jako akviziční kanál, ale drží si 
retention ve vlastním produktu. OpenAI Apps SDK dokonce explicitně mluví o dosažení uživatelů přímo v okamžiku potřeby uvnitř ChatGPT, což zvyšuje atraktivitu této strategie. [30] 
Praktický backlog integrací shrnuje následující tabulka.  
Integrace  Co přesně vybudovat  Proč je to důležité  Priorita 
V271 MCP Gateway  Read-only a action -safe 
konektory pro dokumenty, 
agenty, receipts  Interoperabilita bez 
vendor lock -in P0 
Research object exchange  Import/export research packů, citací, úloh a artefaktů  V271 jako systém záznamu pro AI práci  P0 
Consent ledger  Scope -based paměť, audit 
souhlasů, mazání a export  Důvěra, GDPR a budoucí identity bridge  P0 
Agent card schema  Manifesty pro schopnosti, oprávnění, ceny, reputaci  Základ budoucího federovaného store  P1 
Receipts a provenance  Záznam „co agent udělal, s jakými daty a kdy“  Bezpečnost, reklamace, auditovatelnost  P1 
Native -to-ChatGPT 
app presence  Vybrané surfaces V271 publikovat jako app experience  Akviziční distribuční kanál P1 
Credential wallet  Přenositelné identity a ověřitelné atributy Dlouhodobý most k reputaci a důvěře  P2 
Tato prioritizace vychází z veřejných trendů OpenAI a širšího ekosystému; konkrétní technická proveditelnost každé položky musí být před realizací ověřena proti aktuálním API 
a produktovým podmínkám. [31]  
Produktová roadmapa V271 zarovnaná s pravděpodobným vývojem 
ChatGPT  
Roadmapu je vhodné postavit proti pravděpodobným milníkům ekosystému, ne proti 
nereálné ambici „stihnout všechno“. Přesná roadmapa OpenAI na roky 2027 –2035 veřejně 
neexistuje, takže níže uvedené časování je zarovnání na trendy , nikoli synchronizace s 
neveřejným plánem OpenAI. Jako pracovní hypotézy beru, že: v příštích dvou letech 
poroste význam apps/agents/konektorů; v horizontu 3 –5 let se ustálí standardy tool 
interoperability a hybridní inference; v horizontu 6 –10 let se sjednotí kontinuita, identita, 
multimodální presence a simulation -first planning. Tato hypotéza vychází z dnešních 
veřejných signálů OpenAI, Google/DeepMind, Anthropic, Apple a Microsoft. [32] 
V této části zacházím s V271 jako s B2C produktem s ambicí být osobní AI shell. 
Předpokládám interní baseline: consumer chat, knowledge/library vrstva, agentní katalog, nějaká forma hlasu nebo avataru a více distribučních surfaces. Pokud je tato baseline v reálu užší, roadmapa se má zúžit; pokud je širší, roadmapa se má dál prioritizovat. Není smysluplné stavět vše naráz. Nejdříve je potřeba vybudovat vrstvu kontinuity, souhlasu, receipts a interoperabilních objektů; bez ní bude každá další „cool feature“ působit křehce. 
Tento závěr je strategická inference, ne veřejně potvrzený požadavek OpenAI.  [33] 