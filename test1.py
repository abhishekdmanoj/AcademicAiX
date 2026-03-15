import pickle
with open('vector_store/metadata_chat.pkl', 'rb') as f:
    meta = pickle.load(f)
ktu = [m for m in meta if 'kalam' in m.get('college','').lower()]
if ktu:
    print('College:', ktu[0]['college'])
    print('Program:', ktu[0]['program'])
else:
    print('KTU not found in chat index')
