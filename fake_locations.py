import hashlib
import random
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stores.db")

STATIC_CONVENIENCE_STORES = [
    # Tainan (台南)
    {"name": "7-11 成功門市", "address": "台南市東區大學路1號", "name_en": "7-11 Cheng Kung Store", "address_en": "No. 1, Daxue Rd., East Dist., Tainan City", "lat": 22.9972, "lon": 120.2185},
    {"name": "全家 台南大遠百店", "address": "台南市東區前鋒路210號", "name_en": "FamilyMart Tainan FE21", "address_en": "No. 210, Qianfeng Rd., East Dist., Tainan City", "lat": 22.9959, "lon": 120.2127},
    {"name": "7-11 奇美門市", "address": "台南市永康區中華路901號", "name_en": "7-11 Chi Mei Store", "address_en": "No. 901, Zhonghua Rd., Yongkang Dist., Tainan City", "lat": 23.0189, "lon": 120.2205},
    {"name": "全家 台南成大店", "address": "台南市東區勝利路119號", "name_en": "FamilyMart Tainan NCKU", "address_en": "No. 119, Shengli Rd., East Dist., Tainan City", "lat": 22.9961, "lon": 120.2163},
    {"name": "7-11 安平門市", "address": "台南市安平區安平路792號", "name_en": "7-11 Anping Store", "address_en": "No. 792, Anping Rd., Anping Dist., Tainan City", "lat": 23.0012, "lon": 120.1634},
    {"name": "全家 台南赤崁店", "address": "台南市中西區民族路二段226號", "name_en": "FamilyMart Tainan Chihkan", "address_en": "No. 226, Sec. 2, Minzu Rd., West Central Dist., Tainan City", "lat": 22.9968, "lon": 120.2031},

    # Kaohsiung (高雄)
    {"name": "7-11 建國門市", "address": "高雄市三民區建國二路260號", "name_en": "7-11 Jianguo Store", "address_en": "No. 260, Jianguo 2nd Rd., Sanmin Dist., Kaohsiung City", "lat": 22.6385, "lon": 120.3023},
    {"name": "全家 高雄車站店", "address": "高雄市三民區建國二路320號", "name_en": "FamilyMart Kaohsiung Station", "address_en": "No. 320, Jianguo 2nd Rd., Sanmin Dist., Kaohsiung City", "lat": 22.6391, "lon": 120.3018},
    {"name": "7-11 新高鐵門市", "address": "高雄市左營區高鐵路105號", "name_en": "7-11 Xin Gaotie Store", "address_en": "No. 105, Gaotie Rd., Zuoying Dist., Kaohsiung City", "lat": 22.6876, "lon": 120.3082},
    {"name": "全家 高雄巨蛋店", "address": "高雄市鼓山區裕誠路1156號", "name_en": "FamilyMart Kaohsiung Arena", "address_en": "No. 1156, Yucheng Rd., Gushan Dist., Kaohsiung City", "lat": 22.6668, "lon": 120.3025},
    {"name": "7-11 駁二門市", "address": "高雄市鹽埕區大勇路1號", "name_en": "7-11 Pier-2 Store", "address_en": "No. 1, Dayong Rd., Yancheng Dist., Kaohsiung City", "lat": 22.6202, "lon": 120.2818},
    {"name": "全家 高雄中山店", "address": "高雄市前金區中山二路505號", "name_en": "FamilyMart Kaohsiung Zhongshan", "address_en": "No. 505, Zhongshan 2nd Rd., Qianjin Dist., Kaohsiung City", "lat": 22.6225, "lon": 120.3019},

    # Pingtung (屏東)
    {"name": "7-11 屏東門市", "address": "屏東縣屏東市中山路1號", "name_en": "7-11 Pingtung Store", "address_en": "No. 1, Zhongshan Rd., Pingtung City, Pingtung County", "lat": 22.6718, "lon": 120.4856},
    {"name": "全家 屏東車站店", "address": "屏東縣屏東市公勇路62號", "name_en": "FamilyMart Pingtung Station", "address_en": "No. 62, Gongyong Rd., Pingtung City, Pingtung County", "lat": 22.6689, "lon": 120.4862},
    {"name": "7-11 恆春門市", "address": "屏東縣恆春鎮中山路98號", "name_en": "7-11 Hengchun Store", "address_en": "No. 98, Zhongshan Rd., Hengchun Township, Pingtung County", "lat": 22.0028, "lon": 120.7431},
    {"name": "全家 屏東墾丁店", "address": "屏東縣恆春鎮墾丁路216號", "name_en": "FamilyMart Pingtung Kenting", "address_en": "No. 216, Kenting Rd., Hengchun Township, Pingtung County", "lat": 21.9426, "lon": 120.7981},
    {"name": "7-11 潮州門市", "address": "屏東縣潮州鎮中山路105號", "name_en": "7-11 Chaozhou Store", "address_en": "No. 105, Zhongshan Rd., Chaozhou Township, Pingtung County", "lat": 22.5505, "lon": 120.5422},
    {"name": "全家 屏東東港店", "address": "屏東縣東港鎮中正路一段120號", "name_en": "FamilyMart Pingtung Donggang", "address_en": "No. 120, Sec. 1, Zhongzheng Rd., Donggang Township, Pingtung County", "lat": 22.4678, "lon": 120.4485},
]

def load_stores_from_db():
    if not os.path.exists(DB_PATH):
        return STATIC_CONVENIENCE_STORES
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name, address, name_en, address_en, lat, lon FROM convenience_stores")
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return STATIC_CONVENIENCE_STORES
        return [
            {
                "name": r[0],
                "address": r[1],
                "name_en": r[2],
                "address_en": r[3],
                "lat": float(r[4]),
                "lon": float(r[5])
            }
            for r in rows
        ]
    except Exception:
        return STATIC_CONVENIENCE_STORES

CONVENIENCE_STORES = load_stores_from_db()

def reload_stores():
    global CONVENIENCE_STORES
    CONVENIENCE_STORES = load_stores_from_db()

def set_db_path(path):
    global DB_PATH
    DB_PATH = path

def get_client_location(client_id):
    """
    Load persistent assigned store and client coordinates from SQLite clinical_records.
    """
    if not client_id:
        return {
            "store_name": "",
            "store_address": "",
            "store_lat": 0.0,
            "store_lon": 0.0,
            "client_lat": 0.0,
            "client_lon": 0.0,
        }
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        SELECT assigned_store_id, client_lat, client_lon 
        FROM clinical_records 
        WHERE client_id = ? AND assigned_store_id IS NOT NULL 
        LIMIT 1
        """, (client_id,))
        row = cursor.fetchone()
        if row and row[0]:
            store_id, c_lat, c_lon = row
            cursor.execute("SELECT name, address, lat, lon FROM convenience_stores WHERE osm_id = ?", (store_id,))
            s_row = cursor.fetchone()
            conn.close()
            if s_row:
                return {
                    "store_name": s_row[0],
                    "store_address": s_row[1],
                    "store_lat": round(s_row[2], 5),
                    "store_lon": round(s_row[3], 5),
                    "client_lat": round(c_lat, 5),
                    "client_lon": round(c_lon, 5),
                }
        conn.close()
    except Exception as e:
        print(f"Error reading client location from DB: {e}")
        
    return {
        "store_name": "未分配門市",
        "store_address": "請點選隨機分配進行設定",
        "store_lat": 0.0,
        "store_lon": 0.0,
        "client_lat": 0.0,
        "client_lon": 0.0,
    }
