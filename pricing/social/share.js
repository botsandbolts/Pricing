export default function(component) {
  const {parentElement:root,data}=component;
  const status=root.querySelector('#share-status');
  const report=text=>{status.textContent=text;};
  const copy=async text=>{
    try {await navigator.clipboard.writeText(text);report('Copied. Your next stop: the group chat.');}
    catch {report('Clipboard unavailable here. Select and copy the caption/link shown below.');}
  };
  const link=root.querySelector('#native-link'), image=root.querySelector('#native-image');
  const supported=typeof navigator.share==='function';
  link.disabled=!supported;
  let file=null;
  if(data.png){
    const bytes=Uint8Array.from(atob(data.png),c=>c.charCodeAt(0));
    file=new File([bytes],'car-price-studio.png',{type:'image/png'});
  }
  image.disabled=!(supported&&file&&navigator.canShare?.({files:[file]}));
  if(!supported)report('Device sharing is unavailable here. Use the social links or download your card.');
  link.onclick=async()=>{
    try {await navigator.share({title:data.title,text:data.caption,url:data.url});report('Share action handed to your device.');}
    catch(e){report(e.name==='AbortError'?'Share cancelled. Your car can keep its secrets.':'Sharing blocked here. Use the social links or copy the link.');}
  };
  image.onclick=async()=>{
    try {await navigator.share({files:[file],title:data.title,text:data.caption+'\n'+data.url});report('Image handed to your device.');}
    catch(e){report(e.name==='AbortError'?'Image share cancelled.':'Image sharing blocked here. Download the PNG and attach it to your post.');}
  };
  root.querySelector('#copy-link').onclick=()=>copy(data.url);
  root.querySelector('#copy-caption').onclick=()=>copy(data.caption+'\n'+data.url);
}
