import pkg from 'whatsapp-web.js';
const { Client, LocalAuth } = pkg;
import qrcode from 'qrcode-terminal';
import axios from 'axios';


const client = new Client({
  authStrategy: new LocalAuth(),
  puppeteer: { headless: true } 
  // puppeteer: { headless: false } 
});

client.on('qr', qr => {
  console.log('Scan this QR code with WhatsApp:\n');
  qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
  console.log('WhatsApp client is ready!');
});


client.on('message', async msg => {
  try {
    const chat = await msg.getChat();


    if (chat.isGroup && chat.name.includes("Mess-i-test")) {
      console.log(`[${chat.name}] ${msg.from}: ${msg.body}`);

      
      const res = await axios.post('http://localhost:8000/generate-reply', {
        message: msg.body
      });

      const replyText = res.data["reply"] || "No reply generated.";

      await msg.reply(replyText);
      console.log(`Replied: ${replyText}`);
    }
  } catch (err) {
    console.error("❌ Error:", err.message);
  }
});

client.initialize();
