const admin = require('firebase-admin');

const serviceAccount = require('./key/firebase-readOnly.json');
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
});
const db = admin.firestore();

async function queryPricePrediction(startTimestamp, endTimestamp, compCode) {
    let documents = []; // to save queried results
    try {
        // get data from firestore
        const querySnapshot = await db.collection('classAi_prediction')
            .where('code', '==', compCode)
            .where('datetime', '>=', startTimestamp)
            .where('datetime', '<=', endTimestamp)
            .orderBy('datetime', 'asc')
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

/**
 * 查詢並處理 classAi_prediction 集合中的資料
 * 根據給定的起始時間和結束時間，以及公司代碼進行過濾和排序
 *
 * @param {number} startTimestamp - 查詢的起始時間
 * @param {number} endTimestamp - 查詢的結束時間
 * @param {string} [compCode="GOOG"] - 公司代碼，默認值為 "GOOG"
 * @returns {Promise<{documents: [], count: number}|{documents: [], count: number}>} - 返回查詢結果的 Promise，結果為包含文件 ID 和數據的對象數組。
 * use result.documents to get content like: result.documents['code' / 'price' / 'datetime']
 */
async function queryAndProcess(startTimestamp, endTimestamp, compCode="GOOG") {
    const result = await queryPricePrediction(startTimestamp, endTimestamp, compCode)
    console.log("found data number : ", result.count)
    //console.log(result.documents)
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
    queryAndProcess(1,9999999999).then(r => {
        r.documents.forEach(row=> {
            console.log(row['code']," => ",
                row['price']," => ",
                convertTimestampToDateTime(row['datetime']), "\n")
        });
    });
}

module.exports = {queryAndProcess};
