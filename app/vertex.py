import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file('bhyve-project-398118-7220c36439ce.json')

vertexai.init(project="bhyve-project-398118", location="us-central1", credentials=credentials)
chat_model = ChatModel.from_pretrained("chat-bison")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 256,
    "temperature": 0,
    "top_p": 0,
    "top_k": 0
}
chat = chat_model.start_chat(
    context="""
    another feature of the congress was service of colonial-born educated indians. the  colonial-born  indian  educational  association  was  founded  under  the \\nauspices of the congress. the members consisted mostly of  these educated \\nyouths. they  had  to  pay  a  nominal  subscription. the  association  served  to \\nventilate their needs and grievances, to stimulate thought amongst them, to \\nbring them into touch with indian merchants and also to afford them scope for \\nservice of the community. it was a sort of debating society. the members met \\nregularly and spoke or read papers on different subjects. a small library was \\nalso opened in connection with the association. the  third  feature  of  the  congress  was  propaganda. this  consisted  in \\nacquainting the english in south africa and england and people in india with \\nthe real state of things in natal. with that end in view i wrote two pamphlets. the  first  was an  appeal  to  every  briton  in  south  africa.
    for this purpose it was thought necessary \\nto bring into being a permanent organization. so i consulted sheth abdulla and \\nother friends, and we all decided to have a public organization of a permanent \\ncharacter. to find out a name to be given to the new organization perplexed me sorely. it \\nwas  not  to  identify  itself  with  any  particular  party. the  name  congress,  i \\nknew,  was  in  bad  odour  with  the  conservatives  in  england,  and  yet  the \\ncongress  was  the  very  life  of  india. i  wanted  to  popularize  it  in  natal. it \\nsavoured  of  cowardice  to  hesitate  to  adopt  the  name. therefore,  with  full \\nexplanation of my reasons, i recommended that  the organization should be \\ncalled  the  natal  indian  congress,  and  on  the  22nd  may  the  natal  indian \\ncongress came into being. dada abdullas spacious room was packed to the full on that day. the congress \\nreceived the enthusiastic approval of all present
    www.mkgandhi.org   page 535  an autobiography or my experiments with truth \\n \\n \\n38. congress initiation \\ni must regard my participation in congress proceedings at amritsar as my real \\nentrance into the congress politics. my attendance at the previous congress \\nwas  nothing  more  perhaps  than  an  annual  renewal  of  allegiance  to  the \\ncongress. i never felt on these occasions that i had any other work cut out for \\nme except that of a mere private, nor did i desire more. my experience of amritsar had shown that there were one or two things for \\nwhich perhaps i had some aptitude and which could be useful to the congress. i \\ncould already see that the late lokamanya, the deshabandhu, pandit motilalji \\nand other leaders were pleased with my work in connection with the punjab \\ninquiry. they used to invite me to their informal gatherings where, as i found, \\nresolutions for the subjects committee were conceived.
    
    """)
response = chat.send_message("""Answer Only from Context. Dont Make Up Answers. If You Dont Know Just Tell - I Could Not Find the Answer. Question: Who is Akash Dahad ?""")

print(response) 