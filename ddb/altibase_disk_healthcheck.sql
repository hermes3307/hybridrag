--------------------------------------------------------------------------------
-- Altibase Disk DB Health Check Script
-- Purpose: Collect disk DB usage status and statistics
-- Usage: isql -s localhost -u sys -p manager -f altibase_disk_healthcheck.sql
--------------------------------------------------------------------------------

SET HEADING ON
SET TIMING OFF
SET LINESIZE 200
SET PAGESIZE 1000

SPOOL altibase_disk_healthcheck_result.txt

--------------------------------------------------------------------------------
-- 1. System Basic Information
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '1. System Basic Information' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

-- Version Information
SELECT 
    'ALTIBASE_VERSION' AS INFO_TYPE,
    PRODUCT_VERSION AS VALUE
FROM V$VERSION;

-- System Information
SELECT 
    'INSTANCE_NAME' AS INFO_TYPE,
    DB_NAME AS VALUE
FROM V$DATABASE;

SELECT 
    'START_TIME' AS INFO_TYPE,
    TO_CHAR(STARTUP_TIME, 'YYYY-MM-DD HH24:MI:SS') AS VALUE
FROM V$INSTANCE;

--------------------------------------------------------------------------------
-- 2. Tablespace Status
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '2. Tablespace Status' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    SPACE_NAME,
    CASE SPACE_TYPE 
        WHEN 0 THEN 'MEMORY'
        WHEN 1 THEN 'DISK'
        ELSE 'VOLATILE'
    END AS SPACE_TYPE,
    STATE,
    ROUND(TOTAL_PAGE_COUNT * PAGE_SIZE / 1024 / 1024 / 1024, 2) AS TOTAL_SIZE_GB,
    ROUND(ALLOCATED_PAGE_COUNT * PAGE_SIZE / 1024 / 1024 / 1024, 2) AS ALLOCATED_SIZE_GB,
    ROUND((ALLOCATED_PAGE_COUNT * PAGE_SIZE / 1024 / 1024 / 1024) / 
          NULLIF(TOTAL_PAGE_COUNT * PAGE_SIZE / 1024 / 1024 / 1024, 0) * 100, 2) AS USED_PERCENT,
    AUTOEXTEND_MODE
FROM V$TABLESPACES
ORDER BY SPACE_TYPE, SPACE_NAME;

--------------------------------------------------------------------------------
-- 3. Disk vs Memory DB Size Comparison
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '3. Disk vs Memory DB Size Comparison' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    CASE SPACE_TYPE 
        WHEN 0 THEN 'MEMORY'
        WHEN 1 THEN 'DISK'
        ELSE 'VOLATILE'
    END AS DB_TYPE,
    COUNT(*) AS TABLESPACE_COUNT,
    ROUND(SUM(ALLOCATED_PAGE_COUNT * PAGE_SIZE) / 1024 / 1024 / 1024, 2) AS TOTAL_ALLOCATED_GB,
    ROUND(SUM(TOTAL_PAGE_COUNT * PAGE_SIZE) / 1024 / 1024 / 1024, 2) AS TOTAL_SIZE_GB
FROM V$TABLESPACES
GROUP BY SPACE_TYPE
ORDER BY SPACE_TYPE;

--------------------------------------------------------------------------------
-- 4. Disk Table List (Top 20)
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '4. Disk Table List (Top 20)' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    T.USER_NAME,
    T.TABLE_NAME,
    T.TABLESPACE_NAME,
    T.TABLE_TYPE,
    ROUND(S.DISK_PAGE_CNT * 8 / 1024, 2) AS SIZE_MB,
    T.COLUMN_COUNT,
    S.READ_ROW_COUNT,
    S.INSERT_ROW_COUNT,
    S.UPDATE_ROW_COUNT,
    S.DELETE_ROW_COUNT
FROM SYSTEM_.SYS_TABLES_ T
LEFT OUTER JOIN V$DISKTABLE_STAT S ON T.TABLE_OID = S.TABLE_OID
WHERE T.TABLE_TYPE = 'T'
  AND T.TABLESPACE_ID IN (
      SELECT SPACE_ID FROM V$TABLESPACES WHERE SPACE_TYPE = 1
  )
ORDER BY S.DISK_PAGE_CNT DESC
LIMIT 20;

--------------------------------------------------------------------------------
-- 5. Disk Table Statistics by User
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '5. Disk Table Statistics by User' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    T.USER_NAME,
    COUNT(*) AS TABLE_COUNT,
    ROUND(SUM(S.DISK_PAGE_CNT * 8) / 1024, 2) AS TOTAL_SIZE_MB,
    SUM(S.READ_ROW_COUNT) AS TOTAL_READ_ROWS,
    SUM(S.INSERT_ROW_COUNT) AS TOTAL_INSERT_ROWS,
    SUM(S.UPDATE_ROW_COUNT) AS TOTAL_UPDATE_ROWS,
    SUM(S.DELETE_ROW_COUNT) AS TOTAL_DELETE_ROWS
FROM SYSTEM_.SYS_TABLES_ T
LEFT OUTER JOIN V$DISKTABLE_STAT S ON T.TABLE_OID = S.TABLE_OID
WHERE T.TABLE_TYPE = 'T'
  AND T.TABLESPACE_ID IN (
      SELECT SPACE_ID FROM V$TABLESPACES WHERE SPACE_TYPE = 1
  )
GROUP BY T.USER_NAME
ORDER BY TOTAL_SIZE_MB DESC;

--------------------------------------------------------------------------------
-- 6. Disk Index Status (Top 20)
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '6. Disk Index Status (Top 20)' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    I.USER_NAME,
    I.TABLE_NAME,
    I.INDEX_NAME,
    I.INDEX_TYPE,
    I.IS_UNIQUE,
    ROUND(S.DISK_PAGE_CNT * 8 / 1024, 2) AS SIZE_MB,
    S.READ_COUNT,
    S.INSERT_COUNT,
    S.DELETE_COUNT
FROM SYSTEM_.SYS_INDICES_ I
LEFT OUTER JOIN V$DISKINDEX_STAT S ON I.INDEX_ID = S.INDEX_ID
WHERE I.TABLE_ID IN (
    SELECT TABLE_ID FROM SYSTEM_.SYS_TABLES_ 
    WHERE TABLESPACE_ID IN (SELECT SPACE_ID FROM V$TABLESPACES WHERE SPACE_TYPE = 1)
)
ORDER BY S.DISK_PAGE_CNT DESC
LIMIT 20;

--------------------------------------------------------------------------------
-- 7. Transaction Statistics
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '7. Transaction Statistics' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    'DISK_INSERT_COUNT' AS METRIC,
    SUM(INSERT_ROW_COUNT) AS VALUE
FROM V$DISKTABLE_STAT
UNION ALL
SELECT 
    'DISK_UPDATE_COUNT' AS METRIC,
    SUM(UPDATE_ROW_COUNT) AS VALUE
FROM V$DISKTABLE_STAT
UNION ALL
SELECT 
    'DISK_DELETE_COUNT' AS METRIC,
    SUM(DELETE_ROW_COUNT) AS VALUE
FROM V$DISKTABLE_STAT
UNION ALL
SELECT 
    'DISK_READ_COUNT' AS METRIC,
    SUM(READ_ROW_COUNT) AS VALUE
FROM V$DISKTABLE_STAT;

--------------------------------------------------------------------------------
-- 8. Disk I/O Statistics
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '8. Disk I/O Statistics' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    NAME,
    VALUE
FROM V$SYSSTAT
WHERE NAME LIKE '%DISK%'
   OR NAME LIKE '%disk%'
   OR NAME LIKE '%PAGE%'
ORDER BY NAME;

--------------------------------------------------------------------------------
-- 9. Current Session Information
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '9. Current Session Information' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    ID AS SESSION_ID,
    TRANS_ID,
    DB_USERNAME,
    TASK_STATE,
    QUERY_TIME_LIMIT,
    ACTIVE_FLAG,
    OPENED_STMT_COUNT,
    CLIENT_TYPE,
    CLIENT_APP_INFO
FROM V$SESSION
WHERE TASK_STATE != 'WAITING'
ORDER BY SESSION_ID;

--------------------------------------------------------------------------------
-- 10. Disk Partition Table Information
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '10. Disk Partition Table Information' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    P.USER_NAME,
    P.TABLE_NAME,
    P.PARTITION_NAME,
    P.PARTITION_METHOD,
    P.TABLESPACE_NAME,
    ROUND(S.DISK_PAGE_CNT * 8 / 1024, 2) AS SIZE_MB
FROM SYSTEM_.SYS_TABLE_PARTITIONS_ P
LEFT OUTER JOIN V$DISKTABLE_STAT S ON P.PARTITION_OID = S.TABLE_OID
WHERE P.TABLESPACE_ID IN (
    SELECT SPACE_ID FROM V$TABLESPACES WHERE SPACE_TYPE = 1
)
ORDER BY SIZE_MB DESC
LIMIT 30;

--------------------------------------------------------------------------------
-- 11. Datafile Information
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '11. Datafile Information' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    SPACEID,
    ID AS FILE_ID,
    NAME AS FILE_NAME,
    STATE,
    ROUND(CURRSIZE / 1024 / 1024, 2) AS CURRENT_SIZE_MB,
    ROUND(MAXSIZE / 1024 / 1024, 2) AS MAX_SIZE_MB,
    AUTOEXTEND
FROM V$DATAFILES
ORDER BY SPACEID, FILE_ID;

--------------------------------------------------------------------------------
-- 12. Disk DB Usage Summary
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '12. Disk DB Usage Summary' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    'DISK_DB_RATIO' AS METRIC,
    ROUND(
        (SELECT SUM(ALLOCATED_PAGE_COUNT * PAGE_SIZE) FROM V$TABLESPACES WHERE SPACE_TYPE = 1) /
        NULLIF((SELECT SUM(ALLOCATED_PAGE_COUNT * PAGE_SIZE) FROM V$TABLESPACES), 0) * 100, 2
    ) AS PERCENTAGE
FROM DUAL
UNION ALL
SELECT 
    'MEMORY_DB_RATIO' AS METRIC,
    ROUND(
        (SELECT SUM(ALLOCATED_PAGE_COUNT * PAGE_SIZE) FROM V$TABLESPACES WHERE SPACE_TYPE = 0) /
        NULLIF((SELECT SUM(ALLOCATED_PAGE_COUNT * PAGE_SIZE) FROM V$TABLESPACES), 0) * 100, 2
    ) AS PERCENTAGE
FROM DUAL;

--------------------------------------------------------------------------------
-- 13. Backup Information
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '13. Backup Information' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

-- Backup Level Information
SELECT 
    NAME,
    VALUE
FROM V$PROPERTY
WHERE NAME LIKE '%BACKUP%'
   OR NAME LIKE '%ARCHIVE%'
ORDER BY NAME;

--------------------------------------------------------------------------------
-- 14. Disk DB Related Key Parameters
--------------------------------------------------------------------------------
SELECT '========================================' AS SECTION FROM DUAL;
SELECT '14. Disk DB Related Key Parameters' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SELECT 
    NAME,
    VALUE,
    STOREDCOUNT
FROM V$PROPERTY
WHERE NAME LIKE '%BUFFER%'
   OR NAME LIKE '%DISK%'
   OR NAME LIKE '%FLUSH%'
   OR NAME LIKE '%PAGE%'
ORDER BY NAME;

SELECT '========================================' AS SECTION FROM DUAL;
SELECT 'Health Check Complete' AS SECTION FROM DUAL;
SELECT '========================================' AS SECTION FROM DUAL;

SPOOL OFF

EXIT;
