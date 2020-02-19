DROP TABLE IF EXISTS `Posts`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Posts` (
  `id` int(11) NOT NULL,
  `CreationDate` datetime DEFAULT NULL,
  `Score` int(11) DEFAULT NULL,
  `ViewCount` int(11) DEFAULT NULL,
  `Body` text,
  `LastEditDate` datetime DEFAULT NULL,
  `LastActivityDate` datetime DEFAULT NULL,
  `Title` varchar(250) DEFAULT NULL,
  `Tags` varchar(150) DEFAULT NULL,
  `AnswerCount` int(11) DEFAULT NULL,
  `CommentCount` int(11) DEFAULT NULL,
  `FavoriteCount` int(11) DEFAULT NULL,
  `ClosedDate` datetime DEFAULT NULL,
  `CommunityOwnedDate` datetime DEFAULT NULL,
  `PostTypeId` int(11) NOT NULL,
  `ParentId` int(11) DEFAULT NULL,
  `AcceptedAnswerId` int(11) DEFAULT NULL,
  `OwnerUserId` int(11) DEFAULT NULL,
  `LastEditorUserId` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_Posts_PostTypes1` (`PostTypeId`),
  KEY `fk_Posts_Posts1` (`ParentId`),
  KEY `fk_Posts_Posts2` (`AcceptedAnswerId`),
  KEY `fk_Posts_Users1` (`OwnerUserId`),
  KEY `fk_Posts_CreationDate` (`CreationDate`),
  KEY `score` (`Score`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

