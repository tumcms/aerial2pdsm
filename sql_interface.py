from sqlite3 import Error

sql_create_observations = """ CREATE TABLE IF NOT EXISTS observations(
                                      id integer PRIMARY KEY,
                                      image_id integer NOT NULL,
                                      detected_area_id integer,
                                      i00_x integer ,
                                      i00_y integer,
                                      i11_x integer,
                                      i11_y integer,                                          
                                      geo_hash integer,
                                      qw double,
                                      qx double,
                                      qy double,
                                      qz double,
                                      v1 double,
                                      v2 double,
                                      v3 double,                                                                  
                                      area float
                                  ); """

sql_insert_observation = '''INSERT INTO observations(image_id, i00_x, i00_y, i11_x, i11_y, geo_hash, 
qw, qx, qy, qz, v1, v2, v3, area) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?);'''

sql_create_detected_areas = """CREATE TABLE IF NOT EXISTS detected_areas(
                                      id integer PRIMARY KEY,         
                                      major_observation_id integer NOT NULL,                  
                                      observation_ids TEXT NOT NULL                                                  
                                  ); """

sql_insert_detected_area = """INSERT INTO detected_areas(major_observation_id, observation_ids) VALUES(?, ?);"""

sql_query_observations_by_image_name = """SELECT observations.id, observations.detected_area_id,
    observations.i00_x, observations.i00_y, observations.i11_x, observations.i11_y, images.image_id
    FROM observations INNER JOIN images ON images.image_id = observations.image_id 
    WHERE images.name = ?"""

sql_query_detected_area_by_id = """SELECT * FROM detected_areas WHERE detected_areas.id = ?"""

sql_query_observation_by_id = """SELECT * FROM observations 
    INNER JOIN images ON images.image_id = observations.image_id 
    WHERE observations.id = ?"""

sql_update_observation_detetected_area_id_id = """UPDATE observations SET detected_area_id = ? WHERE observations.id == ?"""


def GetDetectedAreaById(conn, id):
    try:
        return execute_read_query(conn, sql_query_detected_area_by_id, [id])[0]
    except:
        return None


def InsertIntoDetectedAreas(conn, major_observation_id, observation_ids):
    cur = conn.cursor()
    data = [major_observation_id, ",".join(["{}".format(n) for n in observation_ids])]
    cur.execute(sql_insert_detected_area, data)
    return cur.lastrowid


def InsertIntoObservations(conn, image_id, i00, i11, geo_hash, qvec, tvec, area):
    detected_area_row = [image_id, i00.point[0], i00.point[1], i11.point[0],
                         i11.point[1], geo_hash, qvec[0], qvec[1], qvec[2], qvec[3],
                         tvec[0], tvec[1], tvec[2], area]  # 1
    cur = conn.cursor()
    cur.execute(sql_insert_observation, detected_area_row)
    conn.commit()
    return cur.lastrowid


def CreateObservationsTable(conn):
    try:
        c = conn.cursor()
        c.execute(sql_create_observations)
    except Error as e:
        print(e)


def CreateDetectedAreasTable(conn):
    try:
        c = conn.cursor()
        c.execute(sql_create_detected_areas)
    except Error as e:
        print(e)


def execute_read_query(connection, query, data):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query, data)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")


def UpdateObservation_Area(conn, area_id, obs_id):
    cursor = conn.cursor()
    cursor.execute(sql_update_observation_detetected_area_id_id, [area_id, obs_id])


def GetObervationIdByImageName(conn, image_name):
    result = execute_read_query(conn, sql_query_observations_by_image_name, [image_name])
    return result


def GetObservationById(conn, id):
    return execute_read_query(conn, sql_query_observation_by_id, [id])[0]


def GetAllAreas(conn):
    return execute_read_query(conn, """SELECT * FROM detected_areas""", [])
