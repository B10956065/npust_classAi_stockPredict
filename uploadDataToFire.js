const admin = require('firebase-admin');
const serviceAccount = require('./key/firebase-admin.json');
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
});
const db = admin.firestore();

async function uploadToDatabase(data) {
    try {
        const docRef = await db.collection('classAi_prediction').add(data)
        console.log("Document written with ID: ", docRef.id);
        return docRef.id
    } catch (error) {
        console.error("Error adding doc", error)
        return -1
    }
}

function timeStampPrepare(datetime=null) {
    if (datetime == null) {
        const now = new Date()
        return Math.floor(now.getTime() / 1000)
    } else {
        const date = new Date(datetime)
        return Math.floor(date.getTime() / 1000)
    }
}

// main
function collectAndUpload(price, timestamp=timeStampPrepare(), code="GOOG") {
    const data = {
        price: price,
        datetime: timestamp,
        code: code
    }
    console.log(data)
    uploadToDatabase(data).then((id)=>{
        console.log(id)
    });
}

if (require.main === module) {
    const datetime = timeStampPrepare('2024-05-01T03:00:00.0+08:00')
    collectAndUpload(450, timestamp=datetime)
}
