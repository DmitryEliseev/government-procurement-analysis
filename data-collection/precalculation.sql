/*
Preliminary calculations
*/

-- Table creation for storing data on supplier
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='sup_stats' AND xtype='U')
  CREATE TABLE guest.sup_stats (
    SupID INT NOT NULL,
    sup_cntr_num INT,
    sup_running_cntr_num INT,
    sup_good_cntr_num INT,
    sup_fed_cntr_num INT,
    sup_sub_cntr_num INT,
    sup_mun_cntr_num INT,
    sup_cntr_avg_price BIGINT,
    sup_cntr_avg_penalty FLOAT,
    sup_no_pnl_share FLOAT,
    sup_1s_sev FLOAT,
    sup_1s_org_sev FLOAT
    PRIMARY KEY(SupID)
  )

-- Table creation for storing data on customer
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='org_stats' AND xtype='U')
  CREATE TABLE guest.org_stats (
    OrgID INT NOT NULL,
    org_cntr_num INT,
    org_running_cntr_num INT,
    org_good_cntr_num INT,
    org_fed_cntr_num INT,
    org_sub_cntr_num INT,
    org_mun_cntr_num INT,
    org_cntr_avg_price BIGINT,
    org_1s_sev INT,
    org_1s_sup_sev FLOAT,
    PRIMARY KEY(OrgID)
  )

-- Table creation for storing data on OKPD
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='okpd_stats' AND xtype='U')
  CREATE TABLE guest.okpd_stats (
    OkpdID INT NOT NULL PRIMARY KEY,
    code VARCHAR(9),
    cntr_num INT,
    good_cntr_num INT
  )

-- Table creation for storing data on number of contracts of suppliers by OKPD
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='okpd_sup_stats' AND xtype='U')
  CREATE TABLE guest.okpd_sup_stats (
    SupID INT NOT NULL,
    OkpdID INT NOT NULL,
    cntr_num INT,
    PRIMARY KEY (SupID, OkpdID)
  )

-- Table creation for storing data on interection of suppliers and customers
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='sup_org_stats' AND xtype='U')
  CREATE TABLE guest.sup_org_stats (
    SupID INT NOT NULL,
    OrgID INT NOT NULL,
    cntr_num INT,
    PRIMARY KEY (SupID, OrgID)
  )
  
 -- Table creation for storing result of contracts
 IF NOT EXISTS (SELECT * FROM sysobjects WHERE name = 'cntr_stats' AND xtype='U')
   CREATE TABLE guest.cntr_stats (
     CntrID INT NOT NULL PRIMARY KEY,
     result BIT
   )

PRINT('Tables are created')
GO


-- Data collection for table `cntr_stats`
INSERT INTO guest.cntr_stats
SELECT t.cntrID, guest.target(t.cntrID)
FROM
(
  SELECT DISTINCT cntr.ID AS cntrID
  FROM DV.d_OOS_Contracts cntr
  WHERE
    cntr.RefSignDate > guest.utils_get_init_year() AND
    cntr.RefStage IN (3, 4)
)t

PRINT('Data for table `cntr_stats` is collected')
GO

-- Data collection for table `sup_stats` (step I)
INSERT INTO sup_stats (
  SupID, sup_cntr_num, sup_running_cntr_num, sup_good_cntr_num, 
  sup_fed_cntr_num, sup_sub_cntr_num, sup_mun_cntr_num, 
  sup_cntr_avg_price, sup_cntr_avg_penalty
)
SELECT
sup.ID,
guest.sup_num_of_contracts(sup.ID),
guest.sup_num_of_running_contracts(sup.ID),
guest.sup_num_of_good_contracts(sup.ID),
guest.sup_num_of_contracts_lvl(sup.ID, 1),
guest.sup_num_of_contracts_lvl(sup.ID, 2),
NULL,
guest.sup_avg_contract_price(sup.ID),
guest.sup_avg_penalty_share(sup.ID)
FROM DV.d_OOS_Suppliers AS sup

-- Data collection for table `sup_stats` (step II)
UPDATE sup_stats
SET 
  sup_mun_cntr_num = sup_cntr_num - sup_fed_cntr_num - sup_sub_cntr_num,
  sup_no_pnl_share = guest.sup_no_penalty_cntr_share(supID, sup_cntr_num),
  sup_1s_sev = guest.sup_one_side_severance_share(supID, sup_cntr_num),
  sup_1s_org_sev = guest.sup_one_side_org_severance_share(supID, sup_cntr_num)

PRINT('Data for table `sup_stats` is collected')
GO


-- Data collection for table `org_stats` (step I)
INSERT INTO org_stats (
  OrgID, org_cntr_num, org_running_cntr_num, org_good_cntr_num,
  org_fed_cntr_num, org_sub_cntr_num, org_mun_cntr_num, org_cntr_avg_price
)
SELECT
org.ID,
guest.org_num_of_contracts(org.ID),
guest.org_num_of_running_contracts(org.ID),
guest.org_num_of_good_contracts(org.ID),
guest.org_num_of_contracts_lvl(org.ID, 1),
guest.org_num_of_contracts_lvl(org.ID, 2),
NULL,
guest.org_avg_contract_price(org.ID)
FROM DV.d_OOS_Org AS org

-- Data collection for table `org_stats` (step II)
UPDATE org_stats
SET
  org_mun_cntr_num = org_cntr_num - org_fed_cntr_num - org_sub_cntr_num,
  org_1s_sev = guest.org_one_side_severance_share(orgID, org_cntr_num),
  org_1s_sup_sev = guest.org_one_side_supplier_severance_share(orgID, org_cntr_num)

PRINT('Data for table `org_stats` is collected')
GO

-- Data collection for table `okpd_stats` (step I)
INSERT INTO okpd_stats (okpd_stats.OkpdID, okpd_stats.code, okpd_stats.cntr_num)
SELECT okpd.ID, okpd.Code, COUNT(cntr.ID)
FROM 
DV.d_OOS_OKPD2 AS okpd 
INNER JOIN DV.d_OOS_Products AS prods ON prods.RefOKPD2 = okpd.ID
INNER JOIN DV.f_OOS_Product AS prod ON prod.RefProduct = prods.ID
INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = prod.RefContract
WHERE
  cntr.RefStage IN (3, 4) AND
  cntr.RefSignDate > guest.utils_get_init_year()
GROUP BY okpd.ID, okpd.Code

-- Data collection for table `okpd_stats` (step II)
UPDATE okpd_stats
SET okpd_stats.good_cntr_num = t.good_cntr_num
FROM
(
  SELECT okpd.ID AS OkpdID, COUNT(cntr.ID) AS 'good_cntr_num'
  FROM 
  DV.d_OOS_OKPD2 AS okpd 
  INNER JOIN DV.d_OOS_Products AS prods ON prods.RefOKPD2 = okpd.ID
  INNER JOIN DV.f_OOS_Product AS prod ON prod.RefProduct = prods.ID
  INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = prod.RefContract
  INNER JOIN guest.cntr_stats gcs ON cntr.ID = gcs.CntrID
  WHERE
    gcs.result = 0 AND
    cntr.RefSignDate > guest.utils_get_init_year()
  GROUP BY okpd.ID
)t
WHERE t.OkpdID = okpd_stats.OkpdID

PRINT('Data for table `okpd_stats` is collected')
GO

-- Data collection for table `okpd_sup_stats`
INSERT INTO okpd_sup_stats
SELECT t.SupID, t.OkpdID, guest.sup_okpd_cntr_num(t.SupID, t.okpdID)
FROM 
(
  SELECT sup.ID AS SupID, prods.RefOKPD2 AS okpdID
  FROM DV.f_OOS_Product AS prod
  INNER JOIN DV.d_OOS_Suppliers AS sup ON sup.ID = prod.RefSupplier
  INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = prod.RefContract
  INNER JOIN DV.d_OOS_Products AS prods ON prods.ID = prod.RefProduct
  WHERE
    cntr.RefStage in (3, 4) AND
    cntr.RefSignDate > guest.utils_get_init_year()
  GROUP BY sup.ID, prods.RefOKPD2
)t

PRINT('Data for table `okpd_sup_stats` is collected')
GO

-- Data collection for table `sup_org_stats`
INSERT INTO sup_org_stats
SELECT t.supID, t.orgID, guest.sup_org_cntr_num(t.supID, t.orgID)
FROM
(
  SELECT sup.ID AS SupID, org.ID AS OrgID
  FROM DV.f_OOS_Value AS val
  INNER JOIN DV.d_OOS_Suppliers AS sup ON sup.ID = val.RefSupplier
  INNER JOIN DV.d_OOS_Org AS org ON org.ID = val.RefOrg
  INNER JOIN DV.d_OOS_Contracts AS cntr ON cntr.ID = val.RefContract
  WHERE
    cntr.RefStage IN (3, 4) AND
    cntr.RefSignDate > guest.utils_get_init_year()
  GROUP BY sup.ID, org.ID
)t

PRINT('Data for table `sup_org_stats` is collected')
GO