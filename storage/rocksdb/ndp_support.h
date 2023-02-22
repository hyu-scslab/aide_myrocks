#ifndef NDP_SUPPORT_INCLUDED
#define NDP_SUPPORT_INCLUDED

/* Copyright (c) 2000, 2015, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */


/** @file Classes used for SMARTSSD */

#include "./sql_table.h"

namespace myrocks {

struct S3D_JOIN_TAB {
  const char *table_name;
  TABLE* table;
  rocksdb::ColumnFamilyHandleImpl* cfh;
  uint8 idx; // NDP Join Field index
  uint8 op_type; // NDP Filter: op. type
  uint8 attr_type; // NDP Filter: operand type
  uint8 prefix_len; // NDP Filter: used for string match 
                    // (value stored in the operand)
  uint8 target_idx; // NDP Filter: target field
  uint64 operand; // NDP Filter: operand 
} S3D_JOIN_TAB;

typedef uint32 Oid;

typedef enum NDPFilterAttType {
  NDP_FILTER_STRING = 0,
  NDP_FILTER_2B,
  NDP_FILTER_4B,
  NDP_FILTER_8B,
} NDPFilterAttType;

typedef enum NDPFilterOpType {
  NDP_FILTER_EQ = 0,
  NDP_FILTER_GT = 1,
  NDP_FILTER_GTE = 2,
  NDP_FILTER_LT = 3,
  NDP_FILTER_LTE = 4,
  NDP_FILTER_NONE = 5
} NDPFilterOpType;

typedef enum NDPCommandType {
  InvalidCommand = -1,
  NDP_COMMAND_JOIN = 0,
} NDPCommandType;

typedef struct __attribute__((__packed__)) NDPFilter {
  uint8 attr_type;// 0: string, 1: 2B, 2: 4B, 3: 8B
  uint64 operand; // up-to 8B value
  uint8 op_type;  // one of NDPFilterOpType values
  uint8 prefix_len;// used for string match
  uint8 target_idx;// column index for a filter target
} NDPFilter;

typedef struct __attribute__((__packed__)) NDPFilterInfo {
  uint16 num_filter;
  NDPFilter filters[0];
} NDPFilterInfo;

typedef struct __attribute__((__packed__)) NDPColumnInfo {
  uint16 num_att;
  uint16 attlen[0];
  uint16 hash_key_index;
  uint16 hash_key_len;
} NDPColumnInfo;

typedef struct __attribute__((__packed__)) NDPRelationInfoData {
  Oid rel_id;
  uint32 num_segs;
  NDPColumnInfo column_info;
  NDPFilterInfo filter_info;
} NDPRelationInfoData;

typedef struct NDPRelationInfoData* NDPRelationInfo;

struct __attribute__((__packed__)) NDPCommandMessageData {
  NDPCommandType command_type;
  uint32 len;
  uint32 num_relations;
  NDPRelationInfoData relations[0];
} NDPCommandMessageData;

typedef struct NDPCommandMessageData* NDPCommandMessage;

typedef struct NDPRelationFileSegment {
  int data;
  int ctid;
} NDPRelationFileSegment;

typedef struct NDPRelationFileData {
  Oid id;
  uint32 num_segs;
  NDPRelationFileSegment seg[0];
} NDPRelationFileData;

typedef struct NDPRelationFileData* NDPRelationFile;

#define FIXED_HIDDEN_KEY_SIZE (20 + 4)

}// namespace myrocks

#endif /* NDP_SUPPORT_INCLUDED */
