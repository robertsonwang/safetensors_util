import os, re, sys, json
from safetensors_file import SafeTensorsFile

def _need_force_overwrite(output_file:str,cmdLine:dict) -> bool:
    if cmdLine["force_overwrite"]==False:
        if os.path.exists(output_file):
            print(f'output file "{output_file}" already exists, use -f flag to force overwrite',file=sys.stderr)
            return True
    return False

def WriteMetadataToHeader(cmdLine:dict,in_st_file:str,in_json_file:str,output_file:str) -> int:
    if _need_force_overwrite(output_file,cmdLine): return -1

    with open(in_json_file,"rt") as f:
        inmeta=json.load(f)
    if not "__metadata__" in inmeta:
        print(f"file {in_json_file} does not contain a top-level __metadata__ item",file=sys.stderr)
        #json.dump(inmeta,fp=sys.stdout,indent=2)
        return -2
    inmeta=inmeta["__metadata__"] #keep only metadata
    #json.dump(inmeta,fp=sys.stdout,indent=2)

    s=SafeTensorsFile.open_file(in_st_file)
    js=s.get_header()

    if inmeta==[]:
        js.pop("__metadata__",0)
        print("loaded __metadata__ is an empty list, output file will not contain __metadata__ in header")
    else:
        print("adding __metadata__ to header:")
        json.dump(inmeta,fp=sys.stdout,indent=2)
        if isinstance(inmeta,dict):
            for k in inmeta:
                inmeta[k]=str(inmeta[k])
        else:
            inmeta=str(inmeta)
        #js["__metadata__"]=json.dumps(inmeta,ensure_ascii=False)
        js["__metadata__"]=inmeta
        print()

    newhdrbuf=json.dumps(js,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    newhdrlen:int=int(len(newhdrbuf))
    pad:int=((newhdrlen+7)&(~7))-newhdrlen #pad to multiple of 8

    with open(output_file,"wb") as f:
        f.write(int(newhdrlen+pad).to_bytes(8,'little'))
        f.write(newhdrbuf)
        if pad>0: f.write(bytearray([32]*pad))
        i:int=s.copy_data_to_file(f)
    if i==0:
        print(f"file {output_file} saved successfully")
    else:
        print(f"error {i} occurred when writing to file {output_file}")
    return i

def PrintHeader(cmdLine:dict,input_file:str) -> int:
    s=SafeTensorsFile.open_file(input_file,cmdLine['quiet'])
    js=s.get_header()

    # All the .safetensors files I've seen have long key names, and as a result,
    # neither json nor pprint package prints text in very readable format,
    # so we print it ourselves, putting key name & value on one long line.
    # Note the print out is in Python format, not valid JSON format.
    firstKey=True
    print("{")
    for key in js:
        if firstKey:
            firstKey=False
        else:
            print(",")
        json.dump(key,fp=sys.stdout,ensure_ascii=False,separators=(',',':'))
        print(": ",end='')
        json.dump(js[key],fp=sys.stdout,ensure_ascii=False,separators=(',',':'))
    print("\n}")
    return 0

def _ParseMore(d:dict):
    '''Basically try to turn this:

        "ss_dataset_dirs":"{\"abc\": {\"n_repeats\": 2, \"img_count\": 60}}",

    into this:

        "ss_dataset_dirs":{
         "abc":{
          "n_repeats":2,
          "img_count":60
         }
        },

    '''
    for key in d:
        value=d[key]
        #print("+++",key,value,type(value),"+++",sep='|')
        if isinstance(value,str):
            try:
                v2=json.loads(value)
                d[key]=v2
                value=v2
            except json.JSONDecodeError as e:
                pass
        if isinstance(value,dict):
            _ParseMore(value)

def PrintMetadata(cmdLine:dict,input_file:str) -> int:
    with SafeTensorsFile.open_file(input_file,cmdLine['quiet']) as s:
        js=s.get_header()

        if not "__metadata__" in js:
            print("file header does not contain a __metadata__ item",file=sys.stderr)
            return -2

        md=js["__metadata__"]
        if cmdLine['parse_more']:
            _ParseMore(md)
        json.dump({"__metadata__":md},fp=sys.stdout,ensure_ascii=False,separators=(',',':'),indent=1)
    return 0

def HeaderKeysToLists(cmdLine:dict,input_file:str) -> int:
    s=SafeTensorsFile.open_file(input_file,cmdLine['quiet'])
    js=s.get_header()

    _lora_keys:list[tuple(str,bool)]=[] # use list to sort by name
    for key in js:
        if key=='__metadata__': continue
        v=js[key]
        isScalar=False
        if isinstance(v,dict):
            if 'shape' in v:
                if 0==len(v['shape']):
                    isScalar=True
        _lora_keys.append((key,isScalar))
    _lora_keys.sort(key=lambda x:x[0])

    def printkeylist(kl):
        firstKey=True
        for key in kl:
            if firstKey: firstKey=False
            else: print(",")
            print(key,end='')
        print()

    print("# use list to keep insertion order")
    print("_lora_keys:list[tuple[str,bool]]=[")
    printkeylist(_lora_keys)
    print("]")

    return 0


def ExtractHeader(cmdLine:dict,input_file:str,output_file:str)->int:
    if _need_force_overwrite(output_file,cmdLine): return -1

    s=SafeTensorsFile.open_file(input_file,parseHeader=False)
    if s.error!=0: return s.error

    hdrbuf=s.hdrbuf
    s.close_file() #close it in case user wants to write back to input_file itself
    with open(output_file,"wb") as fo:
        wn=fo.write(hdrbuf)
        if wn!=len(hdrbuf):
            print(f"write output file failed, tried to write {len(hdrbuf)} bytes, only wrote {wn} bytes",file=sys.stderr)
            return -1
    print(f"raw header saved to file {output_file}")
    return 0


def _CheckLoRA_internal(s:SafeTensorsFile)->int:
    import lora_keys_sd15 as lora_keys
    js=s.get_header()
    set_scalar=set()
    set_nonscalar=set()
    for x in lora_keys._lora_keys:
        if x[1]==True: set_scalar.add(x[0])
        else: set_nonscalar.add(x[0])

    bad_unknowns:list[str]=[] # unrecognized keys
    bad_scalars:list[str]=[] #bad scalar
    bad_nonscalars:list[str]=[] #bad nonscalar
    for key in js:
        if key in set_nonscalar:
            if js[key]['shape']==[]: bad_nonscalars.append(key)
            set_nonscalar.remove(key)
        elif key in set_scalar:
            if js[key]['shape']!=[]: bad_scalars.append(key)
            set_scalar.remove(key)
        else:
            if "__metadata__"!=key:
                bad_unknowns.append(key)

    hasError=False

    if len(bad_unknowns)!=0:
        print("INFO: unrecognized items:")
        for x in bad_unknowns: print(" ",x)
        #hasError=True

    if len(set_scalar)>0:
        print("missing scalar keys:")
        for x in set_scalar: print(" ",x)
        hasError=True
    if len(set_nonscalar)>0:
        print("missing nonscalar keys:")
        for x in set_nonscalar: print(" ",x)
        hasError=True

    if len(bad_scalars)!=0:
        print("keys expected to be scalar but are nonscalar:")
        for x in bad_scalars: print(" ",x)
        hasError=True

    if len(bad_nonscalars)!=0:
        print("keys expected to be nonscalar but are scalar:")
        for x in bad_nonscalars: print(" ",x)
        hasError=True

    return (1 if hasError else 0)

def CheckLoRA(cmdLine:dict,input_file:str)->int:
    s=SafeTensorsFile.open_file(input_file)
    i:int=_CheckLoRA_internal(s)
    if i==0: print("looks like an OK SD 1.x LoRA file")
    return 0

def ExtractData(cmdLine:dict,input_file:str,key_name:str,output_file:str)->int:
    if _need_force_overwrite(output_file,cmdLine): return -1

    s=SafeTensorsFile.open_file(input_file,cmdLine['quiet'])
    if s.error!=0: return s.error

    bindata=s.load_one_tensor(key_name)
    s.close_file() #close it just in case user wants to write back to input_file itself
    if bindata is None:
        print(f'key "{key_name}" not found in header (key names are case-sensitive)',file=sys.stderr)
        return -1

    with open(output_file,"wb") as fo:
        wn=fo.write(bindata)
        if wn!=len(bindata):
            print(f"write output file failed, tried to write {len(bindata)} bytes, only wrote {wn} bytes",file=sys.stderr)
            return -1
    if cmdLine['quiet']==False: print(f"{key_name} saved to {output_file}, len={wn}")
    return 0

def CheckHeader(cmdLine:dict,input_file:str)->int:
    rv:int=0
    s=SafeTensorsFile.open_file(input_file)
    maxoffset=int(s.st.st_size-8-s.headerlen)
    h=s.get_header()
    for k,v in h.items():
        if k=='__metadata__': continue
        #print(k,v)
        msgs=[]
        if v['data_offsets'][0]>maxoffset or v['data_offsets'][1]>maxoffset:
            msgs.append("data past end of file")
        lenv=int(v['data_offsets'][1]-v['data_offsets'][0])
        items=int(1)
        for i in v['shape']: items*=int(i)

        if v['dtype']=="F16":
            item_size=int(2)
        elif v['dtype']=="F32":
            item_size=int(4)
        elif v['dtype']=="F64":
            item_size=int(8)
        else:
            item_size=int(0)

        if item_size==0:
            if (lenv % items)!=0:
                msgs.append("length not integral multiples of item count")
        else:
            len2=item_size*items
            if len2!=lenv:
                msgs.append(f"length should be {len2}, actual length is {lenv}")

        if len(msgs) > 0:
            print(f"error in f{k}:{v}:")
            for m in msgs:
                print(" * ",m,sep='')
            rv=1

    if rv==0: print("no error found")
    return rv

def DetectLora(cmdLine: dict, input_file: str):
    """
    Try to determine what base architecture a LoRA is meant for using heuristics
    over metadata, parameter names, and adapter shapes. The function returns as
    soon as a confident match is found.

    Heuristic order:
      1. Metadata scan:
         - Look for explicit mentions of SDXL, SD2, SD1.
         - LLM family names: LLaMA, Mistral, Qwen2, Qwen.
         - Flux markers (flux/flux.1/rectified flow/FluxTransformer2DModel/BFL)
      2. Key pattern scan:
         - LLM-style: model.layers.N, self_attn, (q|k|v|o)_proj, mlp.(down|gate|up)_proj.
         - SD-style: unet, text_encoder, lora_te*.
         - Flux-style: lora_transformer_*, transformer_blocks_*_attn_to_(q|k|v|out),
           single_transformer_blocks_*, time_text_embed, context_embedder, x_embedder.
      3. SD refinement:
         - SDXL if a secondary text encoder (text_encoder_2 / te2) is found.
         - Otherwise, infer by text encoder hidden size:
             ~768 → SD1.x
             ~1024 → SD2.x
             ~1280 → SDXL
         - If te1/te2 prefixes exist → SDXL.
      4. Default to "unknown" if no signal matches.

    Returns
    -------
    (label, reasons) : tuple
        label   : string like "sdxl", "sd2.x", "sd1.x", "llama", "mistral",
                  "qwen2", "qwen", "llama/mistral-like", "unknown"
        reasons : dict with "signals" list explaining which hints were matched
    """
    s = SafeTensorsFile.open_file(input_file, cmdLine['quiet'])
    if s.error != 0:
        return "error", {"signals": [f"failed to open file: error {s.error}"]}

    header = s.get_header()
    if header is None:
        return "error", {"signals": ["failed to get header"]}
    metadata = header.get("__metadata__", {})

    reasons = {"signals": []}
    meta = metadata or {}

    # 0) Explicit hints in metadata
    meta_keys = " ".join(k.lower() for k in meta.keys())
    meta_vals = " ".join(str(v).lower() for v in meta.values())
    meta_blob = meta_keys + " " + meta_vals

    def note(sig):
        reasons["signals"].append(sig)

    # Direct metadata hints (common fields used by trainers)
    if any(x in meta_blob for x in ["sdxl", "stable diffusion xl", "text_encoder_2", "te2"]):
        note("metadata mentions SDXL")
        return "sdxl", reasons
    if "sd2" in meta_blob or "sd 2" in meta_blob or "stable diffusion 2" in meta_blob:
        note("metadata mentions SD2")
        return "sd2.x", reasons
    if "sd1" in meta_blob or "sd 1" in meta_blob or "stable diffusion 1" in meta_blob or "sd15" in meta_blob:
        note("metadata mentions SD1")
        return "sd1.x", reasons

    if "llama" in meta_blob:
        note("metadata mentions LLaMA family")
        return "llama", reasons

    if "mistral" in meta_blob:
        note("metadata mentions Mistral family")
        return "mistral", reasons

    if "qwen2" in meta_blob or "qwen-2" in meta_blob:
        note("metadata mentions Qwen2 family")
        return "qwen2", reasons

    if "qwen" in meta_blob:
        note("metadata mentions Qwen (original) family")
        return "qwen", reasons

    # Flux families (Flux.1 / rectified flow / FluxTransformer2DModel / BFL)
    if any(x in meta_blob for x in ["flux.1", "flux1", "flux 1", " flux ", "rectified flow", "fluxtransformer2dmodel", "black-forest-labs", "bfl"]):
        variant = "flux"
        if "schnell" in meta_blob:
            variant = "flux.schnell"
        elif "dev" in meta_blob:
            variant = "flux.dev"
        note("metadata mentions FLUX")
        return variant, reasons

    # -------- Collect keys --------
    keys = [k for k in header.keys() if not k.startswith("__")]

    def has_any(patterns):
        for p in patterns:
            rx = re.compile(p)
            if any(rx.search(k) for k in keys):
                return True
        return False

    def count_matches(patterns):
        rxs = [re.compile(p) for p in patterns]
        return sum(1 for k in keys for rx in rxs if rx.search(k))

    def first_shape_of(regex):
        r = re.compile(regex)
        for k in keys:
            if r.search(k):
                sh = header[k].get("shape", [])
                return tuple(sh)
        return None

    # -------- Flux key-pattern check (before SD/LLM split) --------
    flux_patterns = [
        r"(^|/|_)lora_transformer_",
        r"\btransformer_blocks_\d+_attn_to_(q|k|v|out)",
        r"\bsingle_transformer_blocks_\d+_attn_to_(q|k|v)",
        r"\btime_text_embed",
        r"\bcontext_embedder\b",
        r"\bx_embedder\b",
        r"\bnorm_out_linear\b",
        r"\bproj_out\b",
    ]
    if count_matches(flux_patterns) >= 2:
        # Try variant from obvious filename-style hints in keys (rare), else plain "flux"
        variant = "flux"
        if any("schnell" in k.lower() for k in keys):
            variant = "flux.schnell"
        elif any("dev" in k.lower() for k in keys):
            variant = "flux.dev"
        note("found Flux-style transformer adapter keys")
        return variant, reasons

    # — LLM LoRAs typically look like: "model.layers.0.self_attn.q_proj.lora_down.weight"
    if has_any([r"^model\.layers\.\d+\.", r"\.self_attn\.", r"\.(q|k|v|o)_proj\.", r"\.mlp\.(down|gate|up)_proj\."]):
        note("found LLM-style transformer layer patterns (q_proj/k_proj/etc.)")
        return "llama/mistral-like", reasons

    # SD-family patterns (Kohya/lycoris naming variants)
    # Common UNet/text encoder LoRA key prefixes in SD ecosystem
    is_sd = has_any([
        r"(^|/|_)lora_unet_", r"(^|/|_)lora_te\d?_",
        r"^unet\.", r"^text_encoder(\.|\d|_)", r"text_model\.encoder\.layers",
        r"^transformer\.", r"attentions?\.", r"to_(q|k|v|out)\.lora_(down|up)\.weight",
    ])
    if not is_sd:
        return "unknown", reasons

    # -------- SD refinement --------
    # SDXL almost always includes Text Encoder 1 *and* Text Encoder 2 or explicit "text_encoder_2" blocks
    if has_any([r"text_encoder_2", r"\blora_te2_", r"\bte2\b"]):
        note("found text_encoder_2 / te2 blocks (SDXL hallmark)")
        return "sdxl", reasons

    # If only one text encoder: use hidden size shapes to decide
    # Typical hidden sizes (by common trainers and checkpoints):
    #   SD1.x: CLIP ViT-L/14 (hidden_size ~ 768)
    #   SD2.x: OpenCLIP ViT-H/14 (hidden_size ~ 1024)
    #   SDXL:  Text Encoder 1 ~ 768, Text Encoder 2 ~ 1280 (already caught above)
    #
    # Find a text encoder projection weight to sniff dims (examples):
    #   "...text_model.encoder.layers.X.self_attn.out_proj.lora_down.weight"
    #   Kohya often prefixes with lora_te_...; shapes are [out, in]
    te_shape = (
        first_shape_of(r"text_encoder(\.|_).*lora_(down|A)\.weight")
        or first_shape_of(r"(^|_)lora_te\d?_.+lora_(down|A)\.weight")
    )

    if te_shape:
        out_dim, in_dim = te_shape[:2] if len(te_shape) >= 2 else (None, None)
        note(f"text encoder proj shape hint: {te_shape}")
        # Heuristic thresholds
        if 740 <= (out_dim or 0) <= 800 or 740 <= (in_dim or 0) <= 800:
            note("text encoder hidden size ~768 → SD1.x")
            return "sd1.x", reasons
        if 1000 <= (out_dim or 0) <= 1050 or 1000 <= (in_dim or 0) <= 1050:
            note("text encoder hidden size ~1024 → SD2.x")
            return "sd2.x", reasons
        if 1250 <= (out_dim or 0) <= 1310 or 1250 <= (in_dim or 0) <= 1310:
            note("text encoder hidden size ~1280 (often TE2) → SDXL")
            return "sdxl", reasons

    # If text encoder shapes were inconclusive, look for dual-text encoder naming used by some trainers:
    if has_any([r"\blora_te1_", r"\blora_te2_"]):
        note("saw te1/te2 prefixes → SDXL")
        return "sdxl", reasons

    # Default SD bucket
    return "unknown", reasons
