const admin = require('firebase-admin');

const serviceAccount = require('./key/firebase-readOnly.json');
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
});
const db = admin.firestore();

async function queryPricePrediction(startTime, endTime, code) {
    let documents = []; // 用于存储查询到的文档数据
    try {
        // 向Firestore发起查询
        const querySnapshot = await db.collection('classAi_prediction')
            .get()

        querySnapshot.forEach(doc => {
            documents.push(doc.data());
        });
        return { documents: documents, count: documents.length }
    } catch (error) {
        console.error("Error querying documents: ", error);
        return { documents: [], count: -1 };
    }
}

//
async function queryAndProcess() {
    const result = await queryPricePrediction(0, 0, "GOOG")
    console.log("found data number : ", result.count)
    // console.log(result.documents)
    return result;
}

function convertTimestampToDateTime(timestamp) {
    const date = new Date(timestamp*1000);

    const year = date.getFullYear();
    const month = date.getMonth() + 1; // Month is zero-based, so add 1
    const day = date.getDate();
    const hours = date.getHours();
    const minutes = date.getMinutes();
    const seconds = date.getSeconds();

    return `${year}/${month}/${day}`
}



// test
if (require.main === module) {
    queryAndProcess().then(r => {
        r.documents.forEach(row=> {
            console.log(row['code']," => ",
                row['price']," => ",
                convertTimestampToDateTime(row['datetime']['_seconds']), "\n")
        });
    });
}
