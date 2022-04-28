import pymysql as pmq

# Connect with database
db_connect = pmq.connect(host='106.14.238.7', port=3306, user='root', password='123456', database='trace-stream')
cur = db_connect.cursor()

# Insert cluster item ✅
def db_insert_cluster(cluster_id, create_time, cluster_label, cluster_weight):
    sql = "INSERT INTO cluster (cluster_id, create_time, label, weight) VALUES ({0}, '{1}', '{2}', {3});".format(cluster_id, create_time, cluster_label, cluster_weight)
    cur.execute(sql)
    db_connect.commit() 

# Insert cluster items batch
def db_insert_clusters(cluster_items):
    # cluster_items: '(), (), ()'
    sql = "INSERT INTO cluster (cluster_id, create_time, label, weight) VALUES {0};".format(cluster_items)
    cur.execute(sql)
    db_connect.commit()

# Insert trace items batch
def db_insert_traces(trace_items):
    # trace_items: '(), (), ()'
    sql = "INSERT INTO trace (cluster_id, trace_id, labelled) VALUES {0};".format(trace_items)
    cur.execute(sql)
    db_connect.commit()

# Insert trace item ✅
def db_insert_trace(cluster_id, trace_id):
    sql = "INSERT INTO trace (cluster_id, trace_id, labelled) VALUES ({0}, '{1}', {2});".format(cluster_id, trace_id, 0)
    cur.execute(sql)
    db_connect.commit() 

# Find labelled trace id
def db_find_labelled_trace():
    manual_labels_list = list()
    sql = "SELECT trace_id FROM trace WHERE labelled=1;"
    cur.execute(sql)
    results = cur.fetchall()
    for row in results:
        manual_labels_list.append(row[0])
    return manual_labels_list

# Find all cluster labels ✅
def db_find_cluster_labels():
    cluster_labels = dict()
    sql = "SELECT cluster_id, label FROM cluster;"
    cur.execute(sql)
    results = cur.fetchall()
    for row in results:
        cluster_labels[row[0]] = row[1]
    return cluster_labels

# Update cluster weight ✅
def db_update_weight(cluster_id, cluster_weight):
    sql = "UPDATE cluster set weight={0} WHERE cluster_id={1}".format(cluster_weight, cluster_id)
    cur.execute(sql)
    db_connect.commit()

# Update cluster label ✅
def db_update_label(cluster_id, cluster_label):
    sql = "UPDATE cluster set label='{0}' WHERE cluster_id={1}".format(cluster_label, cluster_id)
    cur.execute(sql)
    db_connect.commit()

# Clear tables ✅
def db_delete_clusterid(cluster_id):
    # clear trace table
    sql = "DELETE FROM trace WHERE cluster_id={0};".format(cluster_id)
    cur.execute(sql)
    db_connect.commit()
    # clear cluster table
    sql = "DELETE FROM cluster WHERE cluster_id={0};".format(cluster_id)
    cur.execute(sql)
    db_connect.commit()

# Clear tables all ✅
def db_delete_all():
    # clear trace table
    sql = "delete from trace;"
    cur.execute(sql)
    db_connect.commit()
    # clear cluster table
    sql = "delete from cluster;"
    cur.execute(sql)
    db_connect.commit()

# Clear cluster table
def db_delete_cluster():
    # clear cluster table
    sql = "delete from cluster;"
    cur.execute(sql)
    db_connect.commit()

# Clear trace table
def db_delete_trace():
    # clear trace table
    sql = "delete from trace;"
    cur.execute(sql)
    db_connect.commit()